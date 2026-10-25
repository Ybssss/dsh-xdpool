/**
 * A `role: "tool"` message must reach the wire with its content intact.
 *
 * Regression: `toolResultsOf()` filtered every `role: "tool"` message down to
 * blocks whose `type` is `'tool-result'`. The OpenAI wire — which is what
 * callers actually send — carries the output as a plain string, so the filter
 * matched nothing, both translators fell through to their `continue`, and the
 * tool output vanished. No warning, no error: the model simply never saw it.
 *
 * The failure is invisible from the request side, which is why it survived: the
 * call still succeeds, the model just answers "No result provided".
 */
import { describe, expect, it } from 'vitest'
import { requestBodyFor } from '../src/free-messages.ts'

const CANARY = 'ZZZ-3Z7-CANARY'

/** Build a one-turn conversation ending in a tool result, in `content` shape. */
function bodyWithToolResult(content: unknown): Record<string, unknown> {
  return {
    model: 'x',
    messages: [
      { role: 'user', content: 'run it' },
      { role: 'assistant', content: '', tool_calls: [{ id: 'call_abc', type: 'function', function: { name: 'bash', arguments: '{}' } }] },
      { role: 'tool', tool_call_id: 'call_abc', content },
    ],
  }
}

function translate(wire: 'chat' | 'responses' | 'messages', content: unknown): string {
  const out = requestBodyFor(wire, bodyWithToolResult(content), { modelId: 'x', warnings: [] })
  return JSON.stringify(out)
}

describe('role:"tool" results survive translation', () => {
  // The three shapes a role:"tool" message can arrive in.
  const shapes: [string, unknown][] = [
    ['OpenAI string', CANARY],
    ['OpenAI content-parts', [{ type: 'text', text: CANARY }]],
    ['Anthropic tool-result block', [{ type: 'tool-result', tool_use_id: 'call_abc', content: CANARY }]],
  ]

  for (const [label, content] of shapes) {
    describe(label, () => {
      it('reaches the responses wire', () => {
        const wire = translate('responses', content)
        expect(wire).toContain(CANARY)
      })

      it('reaches the messages wire', () => {
        const wire = translate('messages', content)
        expect(wire).toContain(CANARY)
      })

      it('keeps the tool_call_id', () => {
        for (const target of ['responses', 'messages'] as const) {
          expect(translate(target, content)).toContain('call_abc')
        }
      })
    })
  }
})

describe('tool result framing', () => {
  it('emits a function_call_output on the responses wire', () => {
    const out = requestBodyFor('responses', bodyWithToolResult(CANARY), { modelId: 'x', warnings: [] })
    const items = out.input as Record<string, unknown>[]
    const call = items.find(item => item.type === 'function_call_output')
    expect(call).toBeDefined()
    expect(call?.output).toBe(CANARY)
  })

  it('emits a tool_result turn on the messages wire', () => {
    const out = requestBodyFor('messages', bodyWithToolResult(CANARY), { modelId: 'x', warnings: [] })
    const turns = out.messages as Record<string, unknown>[]
    const flat = JSON.stringify(turns)
    expect(flat).toContain('tool_result')
    expect(flat).toContain(CANARY)
  })

  it('does not invent a placeholder when content is present', () => {
    for (const wire of ['responses', 'messages'] as const) {
      expect(translate(wire, CANARY)).not.toContain('(no output)')
    }
  })
})

describe('degenerate tool results', () => {
  it('still emits a placeholder for genuinely empty content', () => {
    // An empty result is real and must not be confused with the drop bug.
    const out = translate('responses', '')
    expect(out).toContain('(no output)')
  })

  it('drops nothing silently when content is empty on the messages wire', () => {
    const out = translate('messages', '')
    expect(out).toContain('tool_result')
  })
})

describe('the chat wire is unaffected', () => {
  it('passes the tool message through verbatim', () => {
    // `chat` hands the caller's own messages straight to the upstream, so the
    // OpenAI string shape was always valid there. This pins that the fix did not
    // start rewriting a wire that never needed it.
    const out = requestBodyFor('chat', bodyWithToolResult(CANARY), { modelId: 'x', warnings: [] })
    const messages = out.messages as Record<string, unknown>[]
    const tool = messages.find(message => message.role === 'tool')
    expect(tool?.content).toBe(CANARY)
  })
})
