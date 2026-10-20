import assert from 'node:assert/strict'
import test from 'node:test'
import {
  InteractionStream,
  interactionPeer,
  openInteractionEvents,
} from '../../src/host/dsh-runtime/interaction-stream.js'

function ownerPeer(id: string) {
  return {
    id,
    ctx: {},
    signal: new AbortController().signal,
    dispose() {},
  }
}

test('prefers the DSH 0.2.1 browserPeer and falls back to the 0.2.0 operator', () => {
  const browserPeer = ownerPeer('browser')
  const operator = ownerPeer('operator')
  assert.equal(interactionPeer({ browserPeer, operator } as never), browserPeer)
  assert.equal(interactionPeer({ operator } as never), operator)
  assert.equal(interactionPeer({} as never), undefined)
})

test('opens DSH 0.2.1 Remote events with the live owner answer peer', async () => {
  const peer = ownerPeer('owner')
  let receivedPeer: unknown
  let receivedSignal: AbortSignal | undefined
  const wireStream = {
    open: (
      endpoint: string,
      payload: unknown,
      uplink: unknown,
      passedPeer: unknown,
      signal: AbortSignal,
    ) => {
      assert.equal(endpoint, '$events')
      assert.deepEqual(payload, { args: {} })
      assert.equal(typeof (uplink as AsyncIterable<unknown>)[Symbol.asyncIterator], 'function')
      receivedPeer = passedPeer
      receivedSignal = signal
      return (async function* () {
        yield { type: 'ready', clientId: 'client-1' }
        yield { type: 'user-questions/request', eventId: 'event-1' }
      })()
    },
  }
  const signal = new AbortController().signal
  const frames: unknown[] = []
  const stream = await openInteractionEvents(wireStream as never, signal, peer as never)
  for await (const frame of stream) frames.push(frame)
  assert.equal(receivedPeer, peer)
  assert.equal(receivedSignal, signal)
  assert.deepEqual(frames, [
    { type: 'ready', clientId: 'client-1' },
    { type: 'user-questions/request', eventId: 'event-1' },
  ])
})

test('rejects a DSH 0.2.1 five-argument stream without an owner peer', () => {
  const wireStream = {
    open: (
      _endpoint: string,
      _payload: unknown,
      _uplink: unknown,
      _peer: unknown,
      _signal: AbortSignal,
    ) => (async function* () {})(),
  }
  assert.throws(
    () => openInteractionEvents(wireStream as never, new AbortController().signal),
    /local owner Peer/,
  )
})

test('passes the DSH 0.2.0 operator peer through the five-argument signature', async () => {
  const peer = ownerPeer('operator')
  let receivedPeer: unknown = 'unset'
  const wireStream = {
    open: (
      _endpoint: string,
      _payload: unknown,
      _uplink: unknown,
      passedPeer: unknown,
      _signal: AbortSignal,
    ) => {
      receivedPeer = passedPeer
      return (async function* () {
        yield { type: 'ready', clientId: 'client-2' }
      })()
    },
  }
  const frames: unknown[] = []
  const stream = await openInteractionEvents(
    wireStream as never,
    new AbortController().signal,
    peer as never,
  )
  for await (const frame of stream) frames.push(frame)
  assert.equal(receivedPeer, peer)
  assert.deepEqual(frames, [{ type: 'ready', clientId: 'client-2' }])
})

test('keeps the DSH 0.1.5 three-argument signature working', async () => {
  let receivedSignal: AbortSignal | undefined
  const wireStream = {
    open: (_endpoint: string, _payload: unknown, signal: AbortSignal) => {
      receivedSignal = signal
      return (async function* () {
        yield { type: 'ready', clientId: 'client-3' }
      })()
    },
  }
  const signal = new AbortController().signal
  const frames: unknown[] = []
  const stream = await openInteractionEvents(wireStream as never, signal)
  for await (const frame of stream) frames.push(frame)
  assert.equal(receivedSignal, signal)
  assert.deepEqual(frames, [{ type: 'ready', clientId: 'client-3' }])
})

test('replies through the DSH 0.2.1 Fetch handler with the owner peer', async () => {
  const peer = ownerPeer('browser')
  let receivedPeer: unknown
  let receivedPath: string | undefined
  const handler = {
    async fetch(request: Request, passedPeer?: unknown) {
      receivedPath = new URL(request.url).pathname
      receivedPeer = passedPeer
      const body = await request.json() as { rpcId: string }
      return new Response(JSON.stringify({
        type: 'server-response',
        rpcId: body.rpcId,
        result: { ok: true },
      }), { status: 200, headers: { 'content-type': 'application/json' } })
    },
  }
  const stream = Object.create(InteractionStream.prototype) as InteractionStream
  Object.assign(stream, {
    activeContext: {
      connection: {
        browserPeer: peer,
        createSharedFetchHandler: () => handler,
      },
    },
    clientId: 'client-1',
  })
  await stream.reply('event-1', { kind: 'next' })
  assert.equal(receivedPath, '/api/$events/result')
  assert.equal(receivedPeer, peer)
})
