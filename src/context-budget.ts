/**
 * Context-window budgeting for the WorkBuddy shim.
 *
 * The upstream answers `context_length_exceeded` (business code 11115) when a
 * request overruns the model's window. Rather than bouncing that back to the
 * user as a dead turn, the shim compacts the conversation on the fly:
 *
 *  1. estimate the prompt cost locally (cheap, no round trip);
 *  2. drop the oldest turns while keeping `system` + the newest exchange;
 *  3. if that still overruns, ask the model itself to summarise the middle of
 *     the conversation and splice that summary back in as a system message.
 *
 * Everything here is pure and synchronous-free except `summarizeMessages`,
 * which the caller drives through an injected chat function so this module
 * stays testable without a network.
 *
 * @module dsh-xdpool/context-budget
 */

/** One OpenAI chat message, narrowed to the fields we must preserve. */
export interface ChatMessage {
  role: string
  content: unknown
  [key: string]: unknown
}

/** Rough character-per-token ratio. CJK is ~1 token/char, latin ~1/4. */
const CHARS_PER_TOKEN_LATIN = 4
const CHARS_PER_TOKEN_CJK = 1

/** Fixed per-message overhead the chat template adds (role markers etc.). */
const PER_MESSAGE_TOKEN_OVERHEAD = 4

/** Every image/tool part costs at least this much once decoded. */
const PER_PART_TOKEN_FLOOR = 16

/**
 * Estimate the token cost of one message's `content`.
 *
 * Deliberately conservative (over-estimates) so we compact slightly early
 * rather than discovering the overrun upstream.
 */
export function estimateContentTokens(content: unknown): number {
  if (content === null || content === undefined) return 0
  if (typeof content === 'string') return estimateTextTokens(content)
  if (typeof content === 'number' || typeof content === 'boolean') return PER_PART_TOKEN_FLOOR
  if (Array.isArray(content)) {
    let total = 0
    for (const part of content) total += estimateContentTokens(part)
    return total
  }
  if (typeof content === 'object') {
    const record = content as Record<string, unknown>
    // Multimodal parts carry their payload under `text` / `image_url` / `input`.
    let total = PER_PART_TOKEN_FLOOR
    for (const key of ['text', 'image_url', 'input', 'content']) {
      if (key in record) total += estimateContentTokens(record[key])
    }
    // Tool calls and arbitrary JSON payloads still cost their serialised size.
    if (total === PER_PART_TOKEN_FLOOR) total += estimateTextTokens(safeStringify(record))
    return total
  }
  return 0
}

/** Estimate tokens for a plain string, accounting for CJK density. */
export function estimateTextTokens(text: string): number {
  if (text === '') return 0
  let cjk = 0
  for (const char of text) {
    const code = char.codePointAt(0) ?? 0
    if (isCjk(code)) cjk += 1
  }
  const latin = text.length - cjk
  return Math.ceil(cjk / CHARS_PER_TOKEN_CJK + latin / CHARS_PER_TOKEN_LATIN)
}

function isCjk(code: number): boolean {
  return (
    (code >= 0x3000 && code <= 0x303f) // CJK punctuation
    || (code >= 0x3040 && code <= 0x30ff) // kana
    || (code >= 0x3400 && code <= 0x4dbf) // extension A
    || (code >= 0x4e00 && code <= 0x9fff) // unified ideographs
    || (code >= 0xf900 && code <= 0xfaff) // compatibility ideographs
    || (code >= 0xff00 && code <= 0xffef) // fullwidth forms
    || (code >= 0x20000 && code <= 0x2ebef) // extensions B–F
  )
}

function safeStringify(value: unknown): string {
  try {
    return JSON.stringify(value) ?? ''
  } catch {
    return String(value)
  }
}

/** Estimate the prompt cost of a whole message array. */
export function estimateMessagesTokens(messages: readonly ChatMessage[]): number {
  let total = 0
  for (const message of messages) {
    total += PER_MESSAGE_TOKEN_OVERHEAD
    total += estimateContentTokens(message.content)
    // Tool-call metadata rides alongside `content` and is billed too.
    if (message['tool_calls'] !== undefined) total += estimateContentTokens(message['tool_calls'])
    if (message['name'] !== undefined) total += estimateTextTokens(String(message['name']))
  }
  return total
}

/** True when `role` carries instructions that must survive compaction. */
function isPinnedRole(role: string): boolean {
  return role === 'system' || role === 'developer'
}

export interface CompactOptions {
  /**
   * Token budget for the prompt. The reserve for the model's own output is
   * already subtracted by the caller, so this is the prompt ceiling.
   */
  budget: number
  /** Keep at least this many trailing non-pinned messages verbatim. */
  keepRecent?: number
}

export interface CompactResult {
  /** The compacted message array. */
  messages: ChatMessage[]
  /** Messages removed (oldest first) — the summarisation input. */
  dropped: ChatMessage[]
  /** Estimated tokens after compaction. */
  tokens: number
  /** Whether anything was actually removed. */
  changed: boolean
}

/**
 * Drop the oldest non-pinned messages until the estimate fits `budget`.
 *
 * Pinned (system/developer) messages and the newest `keepRecent` messages are
 * never dropped here — if those alone overrun the budget, the caller must fall
 * back to summarisation or give up.
 */
export function compactMessages(
  messages: readonly ChatMessage[],
  options: CompactOptions,
): CompactResult {
  const keepRecent = Math.max(1, options.keepRecent ?? 4)
  const estimate = estimateMessagesTokens(messages)
  if (estimate <= options.budget) {
    return { messages: [...messages], dropped: [], tokens: estimate, changed: false }
  }

  const pinned: ChatMessage[] = []
  const body: ChatMessage[] = []
  for (const message of messages) {
    if (isPinnedRole(message.role)) pinned.push(message)
    else body.push(message)
  }

  // Always keep the tail; everything older is a drop candidate.
  const keep = Math.min(keepRecent, body.length)
  // A fixed-COUNT tail lands inside a tool group whenever the count is not a
  // multiple of the group size, which for a real agent loop (user + assistant
  // with `tool_calls` + results) is almost always. The seam then leaves results
  // in the tail whose call stayed behind in the head, and the gateway answers
  //   11148 "tool calls and tool results do not match".
  //
  // Walk the seam BACKWARD to the assistant message that owns the group, so the
  // tail carries the call together with all of its results. Walking forward
  // instead would strip results from a call that remains in the head — the
  // mirror image of the same rejection.
  let seam = body.length - keep
  while (seam > 0 && body[seam]?.['role'] === 'tool') seam -= 1
  const tail = body.slice(seam)
  // `head` is the SURVIVING prefix; `dropCount` walks it down from the front, so
  // the dropped set grows while the survivor shrinks. Tracking the dropped set as
  // the survivor list was the original bug: once the prefix emptied, this function
  // reported `changed: false` even though most of the conversation had been cut.
  const head = body.slice(0, seam)
  let dropCount = 0
  let candidate = [...pinned, ...head, ...tail]
  let total = estimateMessagesTokens(candidate)

  while (total > options.budget && dropCount < head.length) {
    dropCount += 1
    // The head is trimmed from the front as well, so this cut needs the same
    // group alignment as the seam: without it the surviving head can open on a
    // result whose call was just dropped.
    const survivor = alignCut(head, dropCount)
    candidate = [...pinned, ...head.slice(survivor), ...tail]
    total = estimateMessagesTokens(candidate)
    if (survivor > dropCount) dropCount = survivor
  }

  // Summarisation input is exactly the prefix trimmed away.
  const dropped = head.slice(0, dropCount)

  return {
    messages: candidate,
    dropped,
    tokens: total,
    changed: dropCount > 0,
  }
}

/**
 * Advance a cut to the next boundary a gateway will accept.
 *
 * A tool group is an assistant message carrying `tool_calls` plus every `tool`
 * message that answers it. Slicing a list by COUNT can land inside one, which
 * leaves a `role: "tool"` message whose parent call was cut away (or the mirror
 * image: a call whose answers were cut away). The gateway answers either with
 *
 *   11148 "tool calls and tool results do not match"
 *
 * so the cut has to snap to a group boundary rather than to an index. Advancing
 * past a leading tool message is sufficient for BOTH directions: a surviving
 * call with no answers is only possible if its answers were dropped, and they
 * can only be dropped from the front, which is exactly the run skipped here.
 *
 * This is needed in two places that used to be one: the head/tail seam inside
 * `compactMessages` (a fixed-count tail always splits a three-message group) and
 * the front cut of `hardTruncate`. Fixing only the front left the seam broken,
 * which is what the 11148 reported.
 */
function alignCut(messages: readonly ChatMessage[], start: number): number {
  let cut = Math.max(0, Math.min(start, messages.length))
  while (cut < messages.length - 1 && messages[cut]?.['role'] === 'tool') cut += 1
  // The "newest message always survives" floor can itself be a tool result whose
  // call was just dropped — the exact 11148 the alignment exists to prevent. When
  // the tail run is nothing but orphaned results, walk BACK to the assistant
  // message that owns them instead of keeping a bare answer. An unsatisfiable
  // budget then yields a tiny but VALID body rather than a rejected one.
  if (cut === messages.length - 1 && messages[cut]?.['role'] === 'tool') {
    let back = cut
    while (back > 0 && messages[back]?.['role'] === 'tool') back -= 1
    cut = back
  }
  return cut
}

/**
 * Drop the oldest messages, including pinned ones, as a last resort.
 *
 * Used when even a summary cannot bring the prompt under budget (for example a
 * single enormous pasted document). The newest message always survives.
 *
 * The cut is aligned to a tool-pairing boundary so the result is always a body
 * the gateway accepts; see {@link alignCut}.
 */
export function hardTruncate(
  messages: readonly ChatMessage[],
  budget: number,
): CompactResult {
  if (messages.length === 0) return { messages: [], dropped: [], tokens: 0, changed: false }
  let start = 0
  let candidate = [...messages]
  let total = estimateMessagesTokens(candidate)
  while (total > budget && start < messages.length - 1) {
    start += 1
    const aligned = alignCut(messages, start)
    candidate = messages.slice(aligned)
    total = estimateMessagesTokens(candidate)
    // Stop advancing the raw cut once alignment alone no longer helps, so the
    // loop cannot spin: `aligned` only grows, and `start` chases it.
    if (aligned > start) start = aligned
  }
  return {
    messages: candidate,
    dropped: messages.slice(0, messages.length - candidate.length),
    tokens: total,
    changed: candidate.length < messages.length,
  }
}

/** Instructions handed to the model when we ask it to compact a conversation. */
export const SUMMARIZE_INSTRUCTION = [
  'You are compacting an ongoing conversation so it can continue without the original history.',
  'Summarise the transcript below into a dense briefing for the next assistant turn.',
  'Preserve, in this order of priority:',
  '1. explicit user requirements, constraints and corrections;',
  '2. decisions already made, and the reasoning behind them;',
  '3. concrete facts: file paths, identifiers, commands, numbers, error messages;',
  '4. unfinished work and the current blocker.',
  'Drop pleasantries, repetition and superseded attempts.',
  'Write the briefing only — no preamble, no markdown fence.',
].join('\n')

/** Render a message array as plain text for the summarisation prompt. */
export function transcriptOf(messages: readonly ChatMessage[]): string {
  const lines: string[] = []
  for (const message of messages) {
    const role = message.role === '' ? 'unknown' : message.role
    lines.push(`### ${role}`)
    lines.push(renderContent(message.content))
    // Keep tool-call structure that the next turn may need to reference.
    if (message['tool_calls'] !== undefined) lines.push(renderContent(message['tool_calls']))
  }
  return lines.join('\n')
}

function renderContent(content: unknown): string {
  if (content === null || content === undefined) return ''
  if (typeof content === 'string') return content
  if (Array.isArray(content)) {
    return content.map(part => renderContent(part)).filter(text => text !== '').join('\n')
  }
  if (typeof content === 'object') {
    const record = content as Record<string, unknown>
    for (const key of ['text', 'content', 'input']) {
      if (typeof record[key] === 'string') return record[key] as string
    }
    if (record['type'] !== undefined && typeof record['type'] === 'string') {
      return `[${record['type']}]`
    }
    return safeStringify(record)
  }
  return String(content)
}

/** Build the synthetic system message that carries a compaction summary. */
export function summaryMessage(summary: string): ChatMessage {
  return {
    role: 'system',
    content: [
      'The earlier part of this conversation was compacted to fit the model context window.',
      'Briefing produced from the dropped turns:',
      '',
      summary.trim(),
    ].join('\n'),
  }
}

export interface SummarizeDeps {
  /**
   * Send a non-streaming completion. Must resolve to the assistant text, or
   * reject so the caller can fall back to plain truncation.
   */
  complete(messages: ChatMessage[], signal?: AbortSignal): Promise<string>
}

export interface SummarizeResult {
  messages: ChatMessage[]
  summary?: string
  /** Why summarisation was skipped, when it was. */
  skipped?: string
  tokens: number
}

/**
 * Compact `messages` to `budget`, summarising the dropped turns when possible.
 *
 * The summary is requested with a *bounded* transcript so the compaction call
 * itself can never overrun the window: if the dropped turns are huge, only the
 * newest slice of them is summarised, and the oldest are noted as elided.
 */
export async function compactWithSummary(
  messages: readonly ChatMessage[],
  options: CompactOptions,
  deps: SummarizeDeps,
  signal?: AbortSignal,
): Promise<SummarizeResult> {
  const first = compactMessages(messages, options)
  if (!first.changed) return { messages: first.messages, tokens: first.tokens, summary: undefined, skipped: undefined }

  // Give the summariser a bounded slice of the transcript.
  const summaryBudget = Math.max(256, Math.floor(options.budget / 4))
  let toSummarize = first.dropped
  let elided = 0
  while (estimateMessagesTokens(toSummarize) > summaryBudget && toSummarize.length > 1) {
    toSummarize = toSummarize.slice(1)
    elided += 1
  }

  let summary: string | undefined
  let skipped: string | undefined
  try {
    const instruction = elided > 0
      ? `${SUMMARIZE_INSTRUCTION}\n\nNote: the ${elided} oldest turn(s) were elided before this transcript.`
      : SUMMARIZE_INSTRUCTION
    const suffix = elided > 0 ? `\n(the ${elided} oldest turn(s) were elided)` : ''
    const request: ChatMessage[] = [
      { role: 'system', content: instruction },
      { role: 'user', content: `${transcriptOf(toSummarize)}${suffix}` },
    ]
    const text = await deps.complete(request, signal)
    if (text.trim() !== '') summary = text.trim()
    else skipped = 'summariser returned an empty summary'
  } catch (error: unknown) {
    skipped = `summarisation failed: ${String(error)}`
  }

  if (summary === undefined) {
    // No summary: fall back to plain truncation, which is still better than
    // surfacing a hard 400 to the user.
    return { messages: first.messages, skipped, tokens: first.tokens }
  }

  const withSummary = injectSummary(first.messages, summary)

  // The summary may itself push us back over budget; hard truncate if so.
  if (estimateMessagesTokens(withSummary) > options.budget) {
    const truncated = hardTruncate(withSummary, options.budget)
    return { messages: truncated.messages, summary, tokens: truncated.tokens }
  }
  return { messages: withSummary, summary, tokens: estimateMessagesTokens(withSummary) }
}

/**
 * Re-insert a summary as: pinned instructions → summary → surviving tail.
 *
 * Order matters. Pinned (system/developer) messages must stay ahead of the
 * summary so that a later synthetic system message can never override the
 * harness's own instructions; the tail follows so the newest exchange is the
 * last thing the model reads.
 *
 * `compacted` is always derived from `original` by `compactMessages`, so the
 * pinned messages it carries are exactly the originals — no need to re-add
 * them from `original`.
 */
function injectSummary(compacted: readonly ChatMessage[], summary: string): ChatMessage[] {
  const pinned: ChatMessage[] = []
  const rest: ChatMessage[] = []
  for (const message of compacted) {
    if (isPinnedRole(message.role)) pinned.push(message)
    else rest.push(message)
  }
  return [...pinned, summaryMessage(summary), ...rest]
}
