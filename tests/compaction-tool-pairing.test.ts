/**
 * Compaction must never orphan a tool message from its `tool_calls`.
 *
 * An OpenAI-compatible gateway rejects a body where a `role: "tool"` message is
 * not preceded by an assistant message carrying the matching `tool_calls` id.
 * Both compaction strategies sliced from the FRONT by count alone, so the
 * surviving window could begin with a tool message whose parent had just been
 * dropped — a structurally invalid request produced by the very step meant to
 * rescue it.
 *
 * This is invisible without a test: the estimator only counts tokens, and a
 * dropped `tool_calls` array is not something the token math notices.
 */
import { describe, expect, it } from 'vitest'
import { compactMessages, hardTruncate } from '../src/context-budget.ts'

type Msg = Record<string, unknown>

/** Every tool id the message list still claims to answer. */
function openCallIds(messages: readonly Msg[]): Set<string> {
  const open = new Set<string>()
  for (const message of messages) {
    if (message.role === 'assistant' && Array.isArray(message.tool_calls)) {
      for (const call of message.tool_calls as Msg[]) {
        if (typeof call.id === 'string') open.add(call.id)
      }
      continue
    }
    if (message.role === 'tool') {
      const id = typeof message.tool_call_id === 'string' ? message.tool_call_id : ''
      open.delete(id)
    }
  }
  return open
}

/** A conversation with `turns` assistant→tool pairs, each with a big payload. */
function conversation(turns: number): Msg[] {
  const out: Msg[] = [{ role: 'system', content: 'you are an agent' }]
  for (let i = 0; i < turns; i += 1) {
    const id = `call_${i}`
    out.push({ role: 'user', content: `step ${i}` })
    out.push({
      role: 'assistant',
      content: '',
      tool_calls: [{ id, type: 'function', function: { name: 'bash', arguments: '{}' } }],
    })
    out.push({ role: 'tool', tool_call_id: id, content: `result ${i} `.repeat(40) })
  }
  return out
}

/**
 * A tool message is valid only when its call id is still open. This asserts the
 * invariant a gateway enforces, without needing a gateway.
 */
function orphans(messages: readonly Msg[]): string[] {
  const open = new Set<string>()
  const bad: string[] = []
  for (const message of messages) {
    if (message.role === 'assistant' && Array.isArray(message.tool_calls)) {
      for (const call of message.tool_calls as Msg[]) {
        if (typeof call.id === 'string') open.add(call.id)
      }
      continue
    }
    if (message.role === 'tool') {
      const id = typeof message.tool_call_id === 'string' ? message.tool_call_id : ''
      if (!open.has(id)) bad.push(id)
      else open.delete(id)
    }
  }
  return bad
}

describe('compactMessages keeps tool pairing', () => {
  it('never leaves a tool message without its tool_calls', () => {
    const messages = conversation(12)
    for (const budget of [400, 800, 1500, 3000, 6000]) {
      const result = compactMessages(messages, { budget })
      expect(orphans(result.messages), `budget ${budget}`).toEqual([])
    }
  })

  it('keeps the leading system message pinned', () => {
    const result = compactMessages(conversation(12), { budget: 800 })
    expect(result.messages[0]?.role).toBe('system')
  })

  it('still reduces the prompt', () => {
    const messages = conversation(12)
    const result = compactMessages(messages, { budget: 1500 })
    expect(result.changed).toBe(true)
    expect(result.messages.length).toBeLessThan(messages.length)
  })
})

describe('hardTruncate keeps tool pairing', () => {
  it('never leaves a tool message without its tool_calls', () => {
    const messages = conversation(12)
    for (const budget of [400, 800, 1500, 3000, 6000]) {
      const result = hardTruncate(messages, budget)
      expect(orphans(result.messages), `budget ${budget}`).toEqual([])
    }
  })

  it('does not strand a tool result at the head of the window', () => {
    // Walk every budget so the cut point lands inside each pairing, which is the
    // arrangement that used to emit a bare tool message first.
    const messages = conversation(12)
    for (let budget = 200; budget <= 4000; budget += 200) {
      const first = hardTruncate(messages, budget).messages[0]
      expect(first?.role, `budget ${budget}`).not.toBe('tool')
    }
  })
})

describe('openCallIds sanity', () => {
  it('treats a paired list as balanced', () => {
    expect([...openCallIds(conversation(3))]).toEqual([])
  })
})
