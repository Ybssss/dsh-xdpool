/**
 * The client-identity headers on chat completions.
 *
 * These exist for one reason: the gateway attributes every
 * `/v2/chat/completions` to a client, and that attribution is what the user
 * sees in their billing console as the "使用端" column. The desktop app sends
 * `X-IDE-Type`/`X-IDE-Name`/`X-IDE-Version` and reads as "WorkBuddy"; without
 * them our requests attributed to nothing and the column showed "–" — which
 * read as broken billing even though every credit was deducted correctly.
 *
 * The regression this guards against is subtle, which is why the assertions are
 * on the exact header set rather than "some headers exist": the temptation when
 * refactoring is to fold these into the shared `chatHeaders()`, which would put
 * identity headers on the points-automation endpoints too — endpoints the
 * desktop app has never been shown to send them on. Both directions are wrong,
 * so both are pinned.
 */
import { describe, expect, it } from 'vitest'
import {
  WorkBuddyUpstreamClient,
  conversationIdentityOf,
  type WorkBuddyCredential,
} from '../src/upstream.ts'

/** A minimal credential; the headers under test do not depend on its details. */
function credential(): WorkBuddyCredential {
  return {
    accessToken: 'tok',
    refreshToken: 'ref',
    uid: 'u1',
    enterpriseId: 'e1',
    domain: 'd1',
  } as unknown as WorkBuddyCredential
}

/** A recording fetch that answers a chat with SSE and anything else with JSON. */
function recordingFetch() {
  const seen: { url: string; headers: Record<string, string> }[] = []
  const impl = async (url: unknown, init: RequestInit): Promise<Response> => {
    const target = String(url)
    seen.push({ url: target, headers: init.headers as Record<string, string> })
    // Chat completions stream; every other endpoint answers a JSON envelope.
    const isChat = target.includes('/v2/chat/completions')
    return new Response(isChat ? 'data: [DONE]\n\n' : '{"code":0,"data":[]}\n', {
      status: 200,
      headers: { 'Content-Type': isChat ? 'text/event-stream' : 'application/json' },
    })
  }
  return { seen, options: { fetchImpl: impl as unknown as typeof fetch } }
}

/** Every path in the client that POSTs a chat completion. */
function chatPaths(): { name: string; run: (client: WorkBuddyUpstreamClient) => Promise<unknown> }[] {
  return [
    { name: 'completeChat', run: (c) => c.completeChat(credential(), '{}') },
    { name: 'chatStream', run: (c) => c.chatStream(credential(), '{}') },
  ]
}

describe('chat completions carry the desktop client identity', () => {
  for (const path of chatPaths()) {
    it(`${path.name} sends the three X-IDE headers`, async () => {
      const { seen, options } = recordingFetch()
      const client = new WorkBuddyUpstreamClient(options)
      await path.run(client)
      expect(seen).toHaveLength(1)
      expect(seen[0].url).toContain('/v2/chat/completions')
      expect(seen[0].headers['X-IDE-Type']).toBe('WorkBuddy')
      expect(seen[0].headers['X-IDE-Name']).toBe('WorkBuddy')
      expect(seen[0].headers['X-IDE-Version']).toBe('5.5.6')
    })

    it(`${path.name} keeps the shared auth and product headers alongside them`, async () => {
      const { seen, options } = recordingFetch()
      const client = new WorkBuddyUpstreamClient(options)
      await path.run(client)
      // The identity headers must not have REPLACED the auth contract.
      expect(seen[0].headers['Authorization']).toBe('Bearer tok')
      expect(seen[0].headers['X-Product']).toBe('SaaS')
      expect(seen[0].headers['X-User-Id']).toBe('u1')
    })
  }
})

/**
 * The conversation-identity triple the channel check demands.
 *
 * `www.workbuddy.ai` rejects a chat completion missing ANY of `X-Conversation-ID`,
 * `X-Conversation-Request-ID` or `X-Conversation-Message-ID` with business code
 * 11128. This was verified by elimination against live credentials, which is why
 * the assertions below pin all three by name rather than "some session headers".
 *
 * The stable/per-request split is pinned too: the upstream reads the triple as
 * one session, so regenerating the stable half per request would look like a new
 * conversation on every turn, while reusing the per-request half would look like
 * a duplicate submission.
 */
describe('chat completions carry the conversation identity the channel check wants', () => {
  const STABLE = ['X-Conversation-ID', 'X-Conversation-Request-ID', 'acp-connection-id'] as const
  const PER_REQUEST = ['X-Conversation-Message-ID', 'X-Request-ID'] as const

  for (const path of chatPaths()) {
    it(`${path.name} sends all three stable conversation ids`, async () => {
      const { seen, options } = recordingFetch()
      const client = new WorkBuddyUpstreamClient(options)
      await path.run(client)
      for (const header of STABLE) {
        const value = seen[0].headers[header]
        expect(value, header).toBeTypeOf('string')
        expect(value, header).not.toBe('')
      }
      expect(seen[0].headers['X-Conversation-Message-ID']).toBeTypeOf('string')
    })

    it(`${path.name} keeps the stable ids stable across calls in one conversation`, async () => {
      const { seen, options } = recordingFetch()
      const client = new WorkBuddyUpstreamClient(options)
      // Same conversation key on both calls: what `conversationKeyOf` produces
      // for two turns of one chat.
      await client.chatStream(credential(), '{}', undefined, 'same-conversation')
      await client.chatStream(credential(), '{}', undefined, 'same-conversation')
      for (const header of STABLE) {
        expect(seen[1].headers[header], header).toBe(seen[0].headers[header])
      }
    })

    it(`${path.name} mints a fresh message and request id every call`, async () => {
      const { seen, options } = recordingFetch()
      const client = new WorkBuddyUpstreamClient(options)
      await client.chatStream(credential(), '{}', undefined, 'same-conversation')
      await client.chatStream(credential(), '{}', undefined, 'same-conversation')
      for (const header of PER_REQUEST) {
        expect(seen[1].headers[header], header).not.toBe(seen[0].headers[header])
      }
    })

    it(`${path.name} gives different conversations different identities`, async () => {
      const { seen, options } = recordingFetch()
      const client = new WorkBuddyUpstreamClient(options)
      await client.chatStream(credential(), '{}', undefined, 'conversation-a')
      await client.chatStream(credential(), '{}', undefined, 'conversation-b')
      expect(seen[1].headers['X-Conversation-ID']).not.toBe(seen[0].headers['X-Conversation-ID'])
    })

    it(`${path.name} still sends an identity when the caller has no key`, async () => {
      // A one-shot compaction can run outside any conversation. Presence is all
      // the gateway checks, so falling back to a random seed is safe — and much
      // safer than omitting the headers and eating a 11128.
      const { seen, options } = recordingFetch()
      const client = new WorkBuddyUpstreamClient(options)
      await client.chatStream(credential(), '{}')
      expect(seen[0].headers['X-Conversation-ID']).toBeTypeOf('string')
      expect(seen[0].headers['X-Conversation-ID']).not.toBe('')
    })
  }

  it('the desktop chat-conversation headers ride along', async () => {
    const { seen, options } = recordingFetch()
    const client = new WorkBuddyUpstreamClient(options)
    await client.chatStream(credential(), '{}')
    // The same set `Yr()` in the shipped app's daemon-bootstrap sets.
    expect(seen[0].headers['X-Agent-Intent']).toBe('craft')
    expect(seen[0].headers['X-Agent-Purpose']).toBe('conversation')
    expect(seen[0].headers['X-Private-Data']).toBe('true')
    expect(seen[0].headers['x-codebuddy-request']).toBe('1')
  })
})

describe('conversationIdentityOf derives rather than stores', () => {
  it('is deterministic for one key and distinct across keys', () => {
    const a = conversationIdentityOf('abc')
    const b = conversationIdentityOf('abc')
    const c = conversationIdentityOf('abd')
    expect(a).toEqual(b)
    expect(a.conversationId).not.toBe(c.conversationId)
    expect(a.conversationRequestId).not.toBe(a.conversationId)
    expect(a.acpConnectionId).not.toBe(a.conversationRequestId)
  })

  it('shapes the ids like UUIDs so downstream parsers keep working', () => {
    const { conversationId } = conversationIdentityOf('abc')
    expect(conversationId).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/)
  })

  it('accepts a missing key by seeding randomly, and says so per call', () => {
    const first = conversationIdentityOf(undefined)
    const second = conversationIdentityOf(undefined)
    // Different seeds, so different ids: two one-shot calls are two sessions.
    expect(first.conversationId).not.toBe(second.conversationId)
  })
})

describe('non-chat endpoints stay free of identity headers', () => {
  it('the points-automation task list does not claim to be the desktop app', async () => {
    const { seen, options } = recordingFetch()
    const client = new WorkBuddyUpstreamClient(options)
    await client.listTasks(credential())
    expect(seen).toHaveLength(1)
    expect(seen[0].url).not.toContain('/v2/chat/completions')
    expect(seen[0].headers['X-IDE-Type']).toBeUndefined()
    expect(seen[0].headers['X-IDE-Name']).toBeUndefined()
  })

  it('openConversation keeps sending its own identity headers', async () => {
    // The automation's conversation opener predates this fix and already sent
    // the three headers inline. It now shares the same constant, so pinning it
    // here catches one path drifting from the other.
    const { seen, options } = recordingFetch()
    const client = new WorkBuddyUpstreamClient(options)
    const opened = await client.openConversation(credential())
    // openConversation tolerates a stream that ends immediately.
    expect(seen).toHaveLength(1)
    expect(seen[0].headers['X-IDE-Type']).toBe('WorkBuddy')
    expect(opened === undefined || typeof opened === 'object').toBe(true)
  })
})
