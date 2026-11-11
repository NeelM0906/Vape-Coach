// vape-coach — Desktop half. Plain ESM, loaded uncompiled (no JSX, no build step).
// Shows roasts in the pet bubble, a status chip, and a coach pane with a live rep counter.
// Data comes from this plugin's own backend (dashboard/plugin_api.py): polled every 3 s
// (the camera watcher is a separate process) plus plugin.vape-coach.* events for instant updates.
import { host, PALETTE_AREA, STATUSBAR_AREAS } from '@hermes/plugin-sdk'
import { useEffect, useState } from 'react'
import { jsx, jsxs } from 'react/jsx-runtime'

const POLL_MS = 3000
// Must match API_VERSION in dashboard/plugin_api.py. Hermes mounts plugin routes when the gateway
// starts, so after an update the old backend keeps answering until Hermes restarts.
const API_VERSION = 2
const RESTART_HINT = 'vape-coach was updated: restart Hermes (or run `hermes gateway restart`) to load the new backend.'
const TONE = { exercise: 'error', pr_threat: 'error', protein_shake: 'wait', meme: 'info', praise: 'info' }
const EMOJI = { exercise: '🏋️', pr_threat: '😈', protein_shake: '🥤', meme: '🤡', praise: '🏆' }

// Tiny shared store so the pane and the chip re-render from one poll loop.
const store = { state: null, latest: null, error: null, listeners: new Set() }
const emit = () => store.listeners.forEach(fn => fn())
function useStore() {
  const [, bump] = useState(0)
  useEffect(() => {
    const fn = () => bump(n => n + 1)
    store.listeners.add(fn)
    return () => store.listeners.delete(fn)
  }, [])
  return store
}

const fmt = s => {
  if (s == null) return '—'
  s = Math.max(0, Math.round(s))
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60
  return h ? `${h}h${String(m).padStart(2, '0')}m` : m ? `${m}m${String(sec).padStart(2, '0')}s` : `${sec}s`
}
const exLabel = e => ({ jumping_jacks: 'jumping jacks', squats: 'squats', pushups: 'push-ups' })[e] || e

function Button({ label, onClick, primary, cooldownMs = 0 }) {
  // cooldownMs: ignore repeat clicks (a double-click on "I just hit it" logged two puffs in testing)
  const [busy, setBusy] = useState(false)
  return jsx('button', {
    type: 'button',
    disabled: busy,
    onClick: async () => {
      if (busy) return
      setBusy(true)
      try { await onClick() } finally {
        if (cooldownMs) setTimeout(() => setBusy(false), cooldownMs)
        else setBusy(false)
      }
    },
    className: `rounded px-2 py-1 text-xs disabled:opacity-50 ${primary ? 'bg-(--ui-accent) text-white' : 'border border-(--ui-border)'}`,
    children: label
  })
}

function Pane({ ctx }) {
  const { state, latest, error } = useStore()
  const st = state?.status
  const [, tick] = useState(0)
  useEffect(() => ctx.setInterval(() => tick(n => n + 1), 1000), [])
  if (error && !st) {
    return jsx('div', {
      className: 'p-3 text-xs text-(--ui-text-tertiary)',
      children: 'vape-coach backend unreachable. Enable the vape-coach plugin (Python half) and restart the gateway.'
    })
  }
  if (!st) return jsx('div', { className: 'p-3 text-xs', children: 'loading…' })
  const sinceFetch = (Date.now() / 1000) - (state.server_ts || 0)
  const wait = Math.max(0, (st.seconds_until_allowed || 0) - sinceFetch)
  const owed = st.open_challenges?.[0]
  const stale = (state.api || 1) < API_VERSION
  const cam = state.camera || {}
  const counting = cam.on && cam.mode === 'reps'
  const repsDone = owed ? owed.done : 0
  const pct = owed ? Math.min(100, Math.round((100 * repsDone) / owed.target)) : 0
  const camLabel = cam.on ? `camera on · ${counting ? 'counting reps' : 'watching'}` : cam.starting ? 'camera starting…' : 'camera off'
  const owedHint = !state.verify_reps ? 'honor system: hit Done when finished'
    : counting ? 'counting on camera… keep your head and arms in frame'
      : cam.on || cam.starting ? 'starting the rep counter… step back from the desk'
        : 'start the camera, step back and do them: I count automatically'
  return jsxs('div', {
    className: 'flex h-full flex-col gap-3 overflow-auto p-3 text-sm',
    children: [
      jsxs('div', { className: 'flex items-center justify-between', children: [
        jsx('div', { className: 'font-medium', children: '💨 Vape Coach' }),
        jsx('div', { className: 'text-[0.6875rem] text-(--ui-text-tertiary)', children: camLabel })
      ] }),
      stale && jsx('div', { className: 'rounded border border-(--ui-border) p-2 text-xs', children: RESTART_HINT }),
      jsxs('div', { className: 'grid grid-cols-2 gap-2 text-xs', children: [
        jsx('div', { children: `phase: ${st.phase}${st.phase === 'coaching' ? ' · day ' + st.day_index : ''}` }),
        jsx('div', { children: `target gap: ${st.target_gap}` }),
        jsx('div', { children: `today: ${st.today?.puffs ?? 0} puffs / ${st.today?.sessions ?? 0} sessions` }),
        jsx('div', { children: `streak: ${st.streak}` }),
        jsx('div', { className: 'col-span-2 text-base font-semibold',
          children: st.phase !== 'coaching' ? 'baseline: just watching today'
            : wait > 0 ? `next hit allowed in ${fmt(wait)}` : 'gap met. still: every minute counts.' })
      ] }),
      owed && jsxs('div', { className: 'rounded border border-(--ui-border) p-2', children: [
        jsx('div', { className: 'text-xs font-medium', children: `owed: ${owed.target} ${exLabel(owed.exercise)}` }),
        jsx('div', { className: 'mt-1 h-2 w-full rounded bg-(--ui-border)', children:
          jsx('div', { className: 'h-2 rounded bg-(--ui-accent)', style: { width: `${pct}%` } }) }),
        jsx('div', { className: 'mt-1 text-2xl font-bold tabular-nums', children: `${repsDone} / ${owed.target}` }),
        jsx('div', { className: 'text-[0.6875rem] text-(--ui-text-tertiary)', children: owedHint })
      ] }),
      latest && jsxs('div', { className: 'rounded bg-(--ui-surface-2) p-3', children: [
        jsx('div', { className: 'text-[0.6875rem] uppercase text-(--ui-text-tertiary)', children: `${EMOJI[latest.kind] || ''} ${latest.kind.replace('_', ' ')}` }),
        jsx('div', { className: 'mt-1 text-base font-semibold leading-snug', children: latest.text })
      ] }),
      jsxs('div', { className: 'flex flex-wrap gap-2', children: [
        jsx(Button, { primary: !cam.on && !cam.starting,
          label: cam.on || cam.starting ? 'Stop camera' : 'Start camera',
          onClick: () => stale ? host.notify({ kind: 'warning', title: 'vape-coach', message: RESTART_HINT })
            : act(ctx, cam.on || cam.starting ? '/camera/stop' : '/camera/start'), cooldownMs: 2000 }),
        jsx(Button, { label: 'I just hit it', onClick: () => act(ctx, '/hit', { puffs: 1 }), cooldownMs: 3000 }),
        owed && !state.verify_reps && jsx(Button, { label: 'Done (honor)', onClick: () => act(ctx, '/done') }),
        jsx(Button, { label: 'Roast me', onClick: async () => {
          const r = await ctx.rest('/roast').catch(() => null)
          if (r?.text) showRoast(ctx, { kind: 'meme', text: r.text })
        } })
      ] })
    ]
  })
}

function Chip() {
  const { state } = useStore()
  const st = state?.status
  if (!st) return null
  const owed = st.open_challenges?.[0]
  const label = owed ? `🏋️ ${owed.target - owed.done} ${exLabel(owed.exercise)}`
    : st.phase === 'coaching' ? `💨 ${st.seconds_until_allowed > 0 ? fmt(st.seconds_until_allowed) : 'ok'}`
      : `💨 ${st.today?.puffs ?? 0} today`
  return jsx('span', { className: 'px-1.5 text-[0.6875rem] text-(--ui-text-tertiary)', title: 'vape-coach', children: label })
}

function showRoast(ctx, iv) {
  store.latest = iv
  emit()
  const text = `${EMOJI[iv.kind] || ''} ${iv.text}`.slice(0, 120)
  if (ctx.pet?.visible?.get?.()) {
    ctx.pet.say(text, { id: 'roast', tone: TONE[iv.kind] || 'info', ttlMs: 15000 })
  } else {
    host.notify({ kind: iv.kind === 'praise' ? 'success' : 'info', title: 'vape-coach', message: iv.text })
  }
}

async function act(ctx, path, body) {
  try {
    const r = await ctx.rest(path, { method: 'POST', body: body || {}, timeoutMs: 15000 })
    if (r && r.ok === false && r.message) host.notify({ kind: 'info', title: 'vape-coach', message: r.message })
  } catch (e) {
    const msg = String(e?.message || e)
    host.notify({ kind: 'warning', title: 'vape-coach', message: /404|not found/i.test(msg) ? RESTART_HINT : `vape-coach: ${msg}` })
  }
  poll(ctx)
}

async function poll(ctx) {
  const since = ctx.storage.get('lastInterventionId', 0)
  try {
    const s = await ctx.rest(`/state?since_id=${since}`)
    store.state = s
    store.error = null
    const fresh = s.interventions || []
    if (fresh.length) {
      ctx.storage.set('lastInterventionId', fresh[fresh.length - 1].id)
      // First load after install: don't replay history, just show the newest.
      if (since > 0 || fresh.length === 1) fresh.slice(-2).forEach(iv => showRoast(ctx, iv))
      else store.latest = fresh[fresh.length - 1]
    }
  } catch (e) {
    store.error = String(e?.message || e)
  }
  emit()
}

export default {
  id: 'vape-coach',
  name: 'Vape Coach',
  defaultEnabled: false,
  register(ctx) {
    ctx.registerMany([
      { id: 'pane', area: 'panes', title: 'vape coach', data: { placement: 'right', width: '300px' },
        render: () => jsx(Pane, { ctx }) },
      { id: 'chip', area: STATUSBAR_AREAS.right, order: 140, render: () => jsx(Chip, {}) },
      { id: 'hit', area: PALETTE_AREA, data: { id: 'vape-coach.hit', label: 'Vape Coach: log a hit',
        keywords: ['vape', 'hit', 'puff'], run: () => act(ctx, '/hit', { puffs: 1 }) } },
      { id: 'roast', area: PALETTE_AREA, data: { id: 'vape-coach.roast', label: 'Vape Coach: roast me',
        keywords: ['vape', 'roast', 'meme'], run: async () => {
          const r = await ctx.rest('/roast').catch(() => null)
          if (r?.text) showRoast(ctx, { kind: 'meme', text: r.text })
        } } }
    ])
    ctx.onEvent('plugin.vape-coach.intervention', ev => {
      const iv = ev?.payload
      if (iv?.text) {
        if (iv.id) ctx.storage.set('lastInterventionId', Math.max(iv.id, ctx.storage.get('lastInterventionId', 0)))
        showRoast(ctx, iv)
      }
      poll(ctx)
    })
    ctx.onEvent('plugin.vape-coach.status', () => poll(ctx))
    poll(ctx)
    ctx.setInterval(() => poll(ctx), POLL_MS)
  }
}
