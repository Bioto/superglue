import { useEffect, useState } from 'react'
import { api } from '../api'

export function ModelPicker({
  selected,
  useDefault,
  hideDefaultOption,
  onChange,
}: {
  selected: string[]
  useDefault: boolean
  hideDefaultOption?: boolean
  onChange: (models: string[], useDefault: boolean) => void
}) {
  const [models, setModels] = useState<string[]>([])
  useEffect(() => {
    api.models().then((result) => setModels(result.data.map((model) => model.id))).catch(() => setModels([]))
  }, [])

  return (
    <div className="model-picker">
      {!hideDefaultOption && (
        <label className="check-row">
          <input type="checkbox" checked={useDefault} onChange={(event) => onChange([], event.target.checked)} />
          Use default models (`openai:*`, `anthropic:*`)
        </label>
      )}
      {(!useDefault || hideDefaultOption) && (
        <div className="model-grid">
          {models.length === 0 ? <span className="muted">No models returned by the gateway.</span> : null}
          {models.map((model) => (
            <label className="check-row" key={model}>
              <input
                type="checkbox"
                checked={selected.includes(model)}
                onChange={(event) => onChange(
                  event.target.checked ? [...selected, model] : selected.filter((value) => value !== model),
                  false,
                )}
              />
              <span className="mono">{model}</span>
            </label>
          ))}
        </div>
      )}
    </div>
  )
}
