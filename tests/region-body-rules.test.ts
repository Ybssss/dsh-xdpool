import { describe, expect, it } from 'vitest'
import { WorkBuddyUpstreamClient } from '../src/upstream.ts'

/**
 * The two body rules that depend on WHICH gateway receives the request.
 *
 * Both were measured against the live gateways, and both cost the user a
 * visible failure before they were handled:
 *
 * - `www.workbuddy.ai` answers `400 / 11128 "first message is not system
 *   prompt"` to a body that opens with a `user` turn. The domestic gateway
 *   accepts the identical body, and uses the very same 11128 code for the
 *   `developer` role and for the content fingerprint — which is exactly why
 *   nothing may branch on the code alone.
 * - Neither gateway accepts a literal `reasoning_effort: "off"`. The adapter
 *   pins `thinkingLevelMap.off` to that literal, and pi-ai emits it whenever
 *   the request carries no explicit level, so "Default" and "Off" produce the
 *   same body.
 */
const client = new WorkBuddyUpstreamClient({ fetchImpl: () => Promise.reject(new Error('unused')) })

function bodyOf(raw: string, region: 'cn' | 'global'): Record<string, unknown> {
  return JSON.parse(client.prepareChatBodyFor(raw, region)) as Record<string, unknown>
}

function withMessages(messages: readonly unknown[]): string {
  return JSON.stringify({ model: 'deepseek-v4.1-flash', messages, stream: true })
}

describe('the international gateway insists on a leading system turn', () => {
  it('prepends one when the body opens with a user message', () => {
    const body = bodyOf(withMessages([{ role: 'user', content: 'hi' }]), 'global')
    const messages = body['messages'] as { role: string; content: unknown }[]
    expect(messages).toHaveLength(2)
    expect(messages[0]?.role).toBe('system')
    expect(typeof messages[0]?.content).toBe('string')
    // The caller's turns keep their order and wording exactly.
    expect(messages[1]).toEqual({ role: 'user', content: 'hi' })
  })

  it('leaves an already-compliant body alone', () => {
    const given = [
      { role: 'system', content: 'you are terse' },
      { role: 'user', content: 'hi' },
    ]
    const body = bodyOf(withMessages(given), 'global')
    expect(body['messages']).toEqual(given)
  })

  it('does not prepend a second system turn when one is merely later in the array', () => {
    // The gateway checks the FIRST message, so a later system turn does not
    // satisfy it — but a leading one must never be doubled either.
    const body = bodyOf(withMessages([{ role: 'user', content: 'hi' }, { role: 'system', content: 'late' }]), 'global')
    const messages = body['messages'] as { role: string }[]
    expect(messages.filter(m => m.role === 'system')).toHaveLength(2)
    expect(messages[0]?.role).toBe('system')
  })

  it('leaves the domestic body untouched, because CN accepts a user opener', () => {
    const given = [{ role: 'user', content: 'hi' }]
    const body = bodyOf(withMessages(given), 'cn')
    expect(body['messages']).toEqual(given)
  })
})

describe('a literal "off" reasoning effort reaches neither gateway', () => {
  it('drops it on the international gateway', () => {
    const raw = JSON.stringify({ messages: [{ role: 'user', content: 'hi' }], reasoning_effort: 'off' })
    expect('reasoning_effort' in bodyOf(raw, 'global')).toBe(false)
  })

  it('drops it on the domestic gateway too', () => {
    const raw = JSON.stringify({ messages: [{ role: 'user', content: 'hi' }], reasoning_effort: 'off' })
    expect('reasoning_effort' in bodyOf(raw, 'cn')).toBe(false)
  })

  it('keeps a real level, since only "off" is rejected', () => {
    const raw = JSON.stringify({ messages: [{ role: 'user', content: 'hi' }], reasoning_effort: 'high' })
    expect(bodyOf(raw, 'global')['reasoning_effort']).toBe('high')
  })

  it('does not rewrite a literal "none", which some models accept', () => {
    // Omission is the only spelling measured good everywhere; inventing a
    // substitute would break the models that do understand the value.
    const raw = JSON.stringify({ messages: [{ role: 'user', content: 'hi' }], reasoning_effort: 'none' })
    expect(bodyOf(raw, 'cn')['reasoning_effort']).toBe('none')
  })
})

describe('the region-aware body still carries the region-blind rules', () => {
  it('forces stream and rewrites the fingerprinted system sentence on both regions', () => {
    for (const region of ['cn', 'global'] as const) {
      const raw = JSON.stringify({
        messages: [{ role: 'system', content: 'You are an AI agent powered by DeepSeek Harness.' }],
        stream: false,
      })
      const body = bodyOf(raw, region)
      expect(body['stream'], region).toBe(true)
      expect(JSON.stringify(body), region).not.toContain('powered by DeepSeek Harness')
    }
  })

  it('returns a non-JSON body verbatim rather than throwing', () => {
    expect(client.prepareChatBodyFor('not json', 'global')).toBe('not json')
  })

  it('threads the region through buildChatBody, which the compaction paths use', () => {
    const body = JSON.parse(
      client.buildChatBody({ stream: true }, [{ role: 'user', content: 'summarise this' }], 'global'),
    ) as { messages: { role: string }[] }
    expect(body.messages[0]?.role).toBe('system')
    expect(body.messages[1]).toEqual({ role: 'user', content: 'summarise this' })
  })

  it('keeps buildChatBody region-blind when no region is passed', () => {
    const body = JSON.parse(
      client.buildChatBody({ stream: true }, [{ role: 'user', content: 'summarise this' }]),
    ) as { messages: { role: string }[] }
    expect(body.messages).toEqual([{ role: 'user', content: 'summarise this' }])
  })
})
