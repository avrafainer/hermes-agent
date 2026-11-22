import type { GatewayEventName } from '@hermes/shared'
// A long, tool-heavy turn whose final text is committed server-side must
// reconcile into the view. Replays the gateway event stream of a turn with
// many tool rounds (interim seals between them, as in the 2026-11-20 incident)
// and asserts the persisted final reply is visible afterwards.
import { act, cleanup } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { chatMessageText } from '@/lib/chat-messages'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'

const SID = 'long-tool-turn-reconcile'
const FINAL = 'Final persisted reply after many tool rounds.'

let stream: MessageStreamHarness

const event = (type: GatewayEventName, timestamp: number, payload: Record<string, unknown> = {}) =>
  act(() => stream.handleEvent({ payload: { ...payload, timestamp }, session_id: SID, type }))

const assistantText = () =>
  (stream.state(SID).messages ?? [])
    .filter(message => message.role === 'assistant')
    .map(chatMessageText)
    .join('\n')

function runTurn(rounds: number, { interim }: { interim: boolean }) {
  event('message.start', 100)
  event('message.delta', 101, { text: 'Working. ', turn_id: 't1' })
  for (let i = 0; i < rounds; i++) {
    const id = `call_${i}`
    event('tool.start', 200 + i * 2, { args: { command: `step ${i}` }, name: 'terminal', tool_id: id })
    if (interim && i % 10 === 0) {
      event('message.interim', 200 + i * 2 + 0.5, { text: `progress ${i}` })
    }
    event('tool.complete', 201 + i * 2, { name: 'terminal', result: 'ok', tool_id: id })
  }
  event('message.delta', 9000, { text: FINAL, turn_id: 't1' })
  event('message.complete', 9001, {
    persisted_turn: { complete: true, final_assistant_row_id: 42 },
    status: 'complete',
    text: FINAL
  })
}

describe('a long tool-heavy turn reconciles its persisted final reply', () => {
  beforeEach(() => {
    stream = renderMessageStream(SID)
  })

  afterEach(() => {
    cleanup()
  })

  // Suspected path: a Stop latch (interrupted=true) that outlives its cancelled
  // execution. message.start is ignored while the latch is set, the turn's
  // deltas and tool rows are dropped, and message.complete takes the
  // interrupted short-circuit, which never merges the persisted final text.
  it('shows the final text when a stale stop latch is still set at turn start', () => {
    stream.states.set(SID, { ...stream.state(SID), interrupted: true })
    runTurn(220, { interim: true })
    expect(assistantText()).toContain(FINAL)
  })

  it('shows the final text after 220 tool rounds with no interim seals', () => {
    runTurn(220, { interim: false })
    expect(assistantText()).toContain(FINAL)
  })

  it('shows the final text after 220 tool rounds with interim seals', () => {
    runTurn(220, { interim: true })
    expect(assistantText()).toContain(FINAL)
  })
})
