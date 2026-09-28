export function ChipRow({ items, limit = 2 }: { items: string[]; limit?: number }) {
  if (items.length === 0) return <span className="muted">—</span>
  const shown = items.slice(0, limit)
  const extra = items.length - shown.length
  return (
    <div className="chip-row" title={items.join(', ')}>
      {shown.map((item) => (
        <span className="chip" key={item}>
          {item}
        </span>
      ))}
      {extra > 0 ? <span className="chip chip-muted">+{extra}</span> : null}
    </div>
  )
}
