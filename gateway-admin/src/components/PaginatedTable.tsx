import { useMemo, useState, type ReactNode } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'

export function PaginatedTable<T>({
  rows,
  searchKeys,
  pageSize = 25,
  searchPlaceholder = 'Search…',
  children,
}: {
  rows: T[]
  searchKeys: (row: T) => string[]
  pageSize?: number
  searchPlaceholder?: string
  children: (pageRows: T[]) => ReactNode
}) {
  const [query, setQuery] = useState('')
  const [page, setPage] = useState(0)

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    if (!q) return rows
    return rows.filter((row) =>
      searchKeys(row).some((v) => v.toLowerCase().includes(q)),
    )
  }, [rows, query, searchKeys])

  const totalPages = Math.max(1, Math.ceil(filtered.length / pageSize))
  const safePage = Math.min(page, totalPages - 1)
  const pageRows = filtered.slice(safePage * pageSize, safePage * pageSize + pageSize)

  return (
    <>
      <div className="flex flex-wrap items-center gap-2">
        <Input
          type="search"
          placeholder={searchPlaceholder}
          value={query}
          onChange={(e) => {
            setQuery(e.target.value)
            setPage(0)
          }}
          className="w-full max-w-xs"
        />
        <span className="text-xs text-muted-foreground">
          {filtered.length} row{filtered.length === 1 ? '' : 's'}
          {filtered.length !== rows.length ? ` (filtered from ${rows.length})` : ''}
        </span>
        {totalPages > 1 && (
          <span className="ml-auto flex items-center gap-2 text-xs text-muted-foreground">
            <Button type="button" variant="outline" size="sm" disabled={safePage === 0} onClick={() => setPage(safePage - 1)}>Prev</Button>
            Page {safePage + 1} / {totalPages}
            <Button type="button" variant="outline" size="sm" disabled={safePage >= totalPages - 1} onClick={() => setPage(safePage + 1)}>Next</Button>
          </span>
        )}
      </div>
      {children(pageRows)}
    </>
  )
}
