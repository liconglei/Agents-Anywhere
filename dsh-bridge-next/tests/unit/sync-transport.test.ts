import test from 'node:test'
import assert from 'node:assert/strict'
import { ConnectorSync } from '../../src/host/dsh-runtime/connection.js'
import { SyncTransportError } from '../../src/host/dsh-runtime/sync.js'
import type { SyncBatch, SyncServiceClient } from '../../src/host/generated/dsh-sync.js'

const batch = {
  subscriptionId: 'sub',
  batchSeq: 1,
  projectionVersion: 3,
  operations: [{ kind: 'notifications', notifications: [] }],
} as unknown as SyncBatch

test('a rejected batch fails the feed instead of letting the next sequence go out', async () => {
  let sent = 0
  const client = {
    ingestBatch: async () => {
      sent += 1
      throw new Error('Unsupported bridge operation: checkpoint.load')
    },
  } as unknown as SyncServiceClient
  const transport = new ConnectorSync(client, 'sub')

  await assert.rejects(
    transport.ingestBatch(batch, new AbortController().signal),
    (error: unknown) => {
      assert.ok(error instanceof SyncTransportError)
      assert.match(String((error as { cause?: unknown }).cause), /Unsupported bridge operation/)
      return true
    },
  )
  assert.equal(sent, 1)
})

test('a rejected checkpoint read is a transport failure too', async () => {
  const client = {
    loadCheckpoint: async () => {
      throw new Error('connector offline')
    },
  } as unknown as SyncServiceClient
  const transport = new ConnectorSync(client, 'sub')

  await assert.rejects(transport.loadCheckpoint('native'), SyncTransportError)
})
