/**
 * Compaction must leave a message list whose tool calls and tool results PAIR UP.
 *
 * The gateway rejects anything else with
 *
 *   11148 "tool calls and tool results do not match,
 *          please start a new conversation and retry"
 *
 * Two distinct ways to violate it, and they need different fixes:
 *
 *   - HEAD: the window begins with a `role: "tool"` message whose parent
 *     `tool_calls` was cut away. Fixed by moving the front cut forward.
 *   - SEAM: the window is assembled from a surviving prefix plus a fixed-count
 *     tail, and the boundary lands INSIDE a call/result group. The result
 *     survives in the middle of the list with no call above it, which the
 *     front-cut fix cannot see.
 *
 * The seam case is the one that actually shipped: `keepRecent` counts messages,
 * and a tool group is three messages, so any cut whose offset is not a multiple
 * of three splits a group.
 */
import { describe, expect, it } from 'vitest'
import { compactMessages, compactWithSummary, hardTruncate } from '../src/context-budget.ts'

type Msg = Record<string, unknown>

/**
 * The invariant a gateway enforces: every tool result must answer a call that is
 * still present ABOVE it, and every call must be answered by a result BELOW it.
 */
function pairingErrors(messages: readonly Msg[]): string[] {
  const errors: string[] = []
  const open = new Set<string>()
  for (let i = 0; i < messages.length; i += 1) {
    const message = messages[i]!
    if (message.role === 'assistant' && Array.isArray(message.tool_calls)) {
      for (const call of message.tool_calls as Msg[]) {
        if (typeof call.id === 'string') open.add(call.id)
      }
      continue
    }
    if (message.role === 'tool') {
      const id = typeof message.tool_call_id === 'string' ? message.tool_call_id : ''
      if (!open.has(id)) errors.push(`orphan result ${id} at ${i}`)
      else open.delete(id)
    }
  }
  for (const id of open) errors.push(`unanswered call ${id}`)
  return errors
}

/**
 * A conversation of assistant→tool pairs, shaped like a real agent loop.
 *
 * Each group is three messages (user, assistant+tool_calls, tool), so a
 * fixed-count tail of 4 always lands mid-group.
 */
function loop(turns: number): Msg[] {
  const out: Msg[] = [{ role: 'system', content: 'you are an agent' }]
  for (let i = 0; i < turns; i += 1) {
    const id = `call_${i}`
    out.push({ role: 'user', content: `step ${i}` })
    out.push({
      role: 'assistant',
      content: '',
      tool_calls: [{ id, type: 'function', function: { name: 'bash', arguments: '{}' } }],
    })
    out.push({ role: 'tool', tool_call_id: id, content: `result ${i} `.repeat(30) })
  }
  return out
}

/** A loop where one assistant turn issues several calls at once. */
function multiCall(turns: number): Msg[] {
  const out: Msg[] = [{ role: 'system', content: 'you are an agent' }]
  for (let i = 0; i < turns; i += 1) {
    const ids = [`a${i}`, `b${i}`, `c${i}`]
    out.push({ role: 'user', content: `step ${i}` })
    out.push({
      role: 'assistant',
      content: '',
      tool_calls: ids.map(id => ({ id, type: 'function', function: { name: 'bash', arguments: '{}' } })),
    })
    for (const id of ids) out.push({ role: 'tool', tool_call_id: id, content: `r ${id} `.repeat(30) })
  }
  return out
}

describe('compactMessages leaves a pairable list', () => {
  it('no seam orphan across every budget', () => {
    const messages = loop(12)
    for (let budget = 300; budget <= 9000; budget += 300) {
      expect(pairingErrors(compactMessages(messages, { budget }).messages), `budget ${budget}`).toEqual([])
    }
  })

  it('no seam orphan with several calls per turn', () => {
    const messages = multiCall(8)
    for (let budget = 300; budget <= 9000; budget += 300) {
      expect(pairingErrors(compactMessages(messages, { budget }).messages), `budget ${budget}`).toEqual([])
    }
  })

  it('no seam orphan at any keepRecent', () => {
    const messages = loop(12)
    for (const keepRecent of [1, 2, 3, 4, 5, 6, 7]) {
      for (const budget of [600, 1200, 2400, 4800]) {
        expect(
          pairingErrors(compactMessages(messages, { budget, keepRecent }).messages),
          `keepRecent ${keepRecent} budget ${budget}`,
        ).toEqual([])
      }
    }
  })

  it('keeps the leading system message', () => {
    expect(compactMessages(loop(12), { budget: 900 }).messages[0]?.role).toBe('system')
  })

  it('still reduces the prompt', () => {
    const messages = loop(12)
    // Budget must be BELOW the conversation's own size, or nothing is dropped and
    // `changed` is honestly false. This loop is ~1438 tokens.
    const result = compactMessages(messages, { budget: 900 })
    expect(result.changed).toBe(true)
    expect(result.messages.length).toBeLessThan(messages.length)
  })
})

describe('hardTruncate leaves a pairable list', () => {
  it('no orphan across every budget', () => {
    const messages = loop(12)
    for (let budget = 200; budget <= 6000; budget += 200) {
      expect(pairingErrors(hardTruncate(messages, budget).messages), `budget ${budget}`).toEqual([])
    }
  })

  it('no orphan with several calls per turn', () => {
    const messages = multiCall(8)
    for (let budget = 200; budget <= 6000; budget += 200) {
      expect(pairingErrors(hardTruncate(messages, budget).messages), `budget ${budget}`).toEqual([])
    }
  })
})

describe('the injected summary is not a tool result', () => {
  it('a summarised compaction still pairs up', () => {
    // The injected summary is a plain system message; it must not be mistaken
    // for a result, and must not sit between a call and its answer.
    const result = compactMessages(loop(10), { budget: 1500 })
    expect(pairingErrors(result.messages)).toEqual([])
  })

  it('survives the summary→truncate composition', async () => {
    // The real recovery path runs compactWithSummary, which injects a summary and
    // may then hard-truncate the result. Both steps must preserve pairing.
    const result = await compactWithSummary(
      loop(12),
      { budget: 700 },
      { complete: async () => 'briefing: the agent ran several shell commands' },
      undefined,
    )
    expect(pairingErrors(result.messages)).toEqual([])
  })
})
