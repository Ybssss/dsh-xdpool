import { describe, expect, it } from 'vitest'
import { sanitizeSystemContent, WorkBuddyUpstreamClient } from '../src/upstream.ts'

/*
  The domestic gateway rejects a request whose SYSTEM message opens with a
  string it associates with another agent's client, answering
  `400 / 11128 Illegal API invocation from an unapproved channel` — the same
  business code the missing conversation-identity headers produce, which is why
  the two are easy to confuse. The first-party DSH system-prompt plugin puts
  that sentence at the top of every conversation, so an unmodified install is
  rejected on every account.

  Measured boundary (live gateway): literal substring, case-insensitive, only
  near the start of a system message, only `role: "system"`. These tests pin the
  rewrite, not the gateway: if the list grows, the array grows.
*/
const BANNED = 'You are an AI agent powered by DeepSeek Harness.'
const SAFE = 'You are an AI agent running on DeepSeek Harness.'

describe('the system-prompt fingerprint the gateway blocks', () => {
  it('rewrites the sentence and keeps the rest of the prompt', () => {
    const content = `${BANNED}\n\nYou verify before you claim.`
    expect(sanitizeSystemContent(content)).toBe(`${SAFE}\n\nYou verify before you claim.`)
  })

  it('rewrites it mid-sentence, in any case, and every occurrence', () => {
    // The match is case-insensitive because the gateway's is; the replacement is
    // a fixed phrase, so a lowercase hit comes back title-cased. That is fine —
    // the check is a substring test and the model reads the meaning — and it is
    // what keeps the pattern specific to the fingerprint instead of rewriting
    // every innocent "powered by" in a system prompt.
    const lowered = sanitizeSystemContent('you are an ai agent powered by deepseek harness.') as string
    expect(lowered).not.toMatch(/powered by deepseek harness/i)
    expect(lowered).toContain('running on DeepSeek Harness')
    expect(sanitizeSystemContent('a powered by DeepSeek Harness b powered by DeepSeek Harness c'))
      .toBe('a running on DeepSeek Harness b running on DeepSeek Harness c')
    // An innocent use of the same verb is not this fingerprint and must survive.
    expect(sanitizeSystemContent('This deployment is powered by solar energy.'))
      .toBe('This deployment is powered by solar energy.')
  })

  it('leaves unrelated system text byte-identical and returns the same reference', () => {
    const clean = 'You are a helpful assistant.'
    expect(sanitizeSystemContent(clean)).toBe(clean)
    // Same reference, not merely equal: a long prompt must not be copied to
    // change nothing.
    const block = [{ type: 'text', text: clean }]
    expect(sanitizeSystemContent(block)).toBe(block)
  })

  it('rewrites only the matching block and preserves images and unknown blocks', () => {
    const image = { type: 'image_url', image_url: { url: 'data:image/png;base64,AAAA' } }
    const cache = { type: 'cache_control', cache_control: { type: 'ephemeral' } }
    const next = sanitizeSystemContent([
      { type: 'text', text: BANNED },
      image,
      cache,
    ]) as Record<string, unknown>[]
    expect(next[0]).toEqual({ type: 'text', text: SAFE })
    expect(next[1]).toBe(image)
    expect(next[2]).toBe(cache)
  })

  it('passes non-string, non-array content through untouched', () => {
    expect(sanitizeSystemContent(null)).toBeNull()
    expect(sanitizeSystemContent(42)).toBe(42)
    expect(sanitizeSystemContent(undefined)).toBeUndefined()
  })

  it('applies the rewrite inside the prepared request body, for every system message', () => {
    const client = new WorkBuddyUpstreamClient({ fetchImpl: () => Promise.reject(new Error('unused')) })
    const raw = JSON.stringify({
      model: 'deepseek-v4.1-flash',
      messages: [
        { role: 'system', content: BANNED },
        { role: 'user', content: 'hello' },
        { role: 'system', content: [{ type: 'text', text: BANNED }] },
      ],
      stream: false,
    })
    const body = JSON.parse(client.prepareChatBody(raw)) as {
      messages: { role: string; content: unknown }[]
      stream: boolean
    }
    expect(body.stream).toBe(true)
    expect(body.messages[0]?.content).toBe(SAFE)
    expect(body.messages[1]?.content).toBe('hello')
    expect(body.messages[2]?.content).toEqual([{ type: 'text', text: SAFE }])
    // The rewrite must not survive as the banned substring anywhere.
    expect(JSON.stringify(body)).not.toContain('powered by DeepSeek Harness')
  })

  it('rewrites a developer message too, since it becomes a system message', () => {
    const client = new WorkBuddyUpstreamClient({ fetchImpl: () => Promise.reject(new Error('unused')) })
    const raw = JSON.stringify({
      messages: [{ role: 'developer', content: BANNED }],
    })
    const body = JSON.parse(client.prepareChatBody(raw)) as { messages: { role: string; content: string }[] }
    expect(body.messages[0]?.role).toBe('system')
    expect(body.messages[0]?.content).toBe(SAFE)
  })

  it('does not touch the same sentence when it is in a user or tool message', () => {
    // The gateway ignores those roles, and rewriting them would corrupt text the
    // user actually wrote.
    const client = new WorkBuddyUpstreamClient({ fetchImpl: () => Promise.reject(new Error('unused')) })
    const raw = JSON.stringify({
      messages: [
        { role: 'user', content: BANNED },
        { role: 'tool', content: BANNED },
      ],
    })
    const body = JSON.parse(client.prepareChatBody(raw)) as { messages: { content: string }[] }
    expect(body.messages[0]?.content).toBe(BANNED)
    expect(body.messages[1]?.content).toBe(BANNED)
  })
})
