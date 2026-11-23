import type { ModelInfo } from './api'

export const CLASS_UK: Record<string, string> = {
  aeroplane: 'літак', bicycle: 'велосипед', bird: 'птах', boat: 'човен', bottle: 'пляшка',
  bus: 'автобус', car: 'автомобіль', cat: 'кіт', chair: 'стілець', cow: 'корова',
  diningtable: 'обідній стіл', dog: 'собака', horse: 'кінь', motorbike: 'мотоцикл', person: 'людина',
  pottedplant: 'рослина в горщику', sheep: 'вівця', sofa: 'диван', train: 'потяг', tvmonitor: 'монітор',
}

export const uk = (label: string) => CLASS_UK[label] ?? label

export const ENCODING_UK: Record<string, string> = { amplitude: 'Амплітудне', angle: 'Кутове', zz: 'ZZ-відображення' }

/** Ukrainian plural: plural(4, ['кубіт', 'кубіти', 'кубітів']) -> 'кубіти' */
export function plural(n: number, [one, few, many]: [string, string, string]) {
  const m10 = n % 10, m100 = n % 100
  return m10 === 1 && m100 !== 11 ? one : m10 >= 2 && m10 <= 4 && (m100 < 12 || m100 > 14) ? few : many
}
export const qubits = (n: number) => `${n} ${plural(n, ['кубіт', 'кубіти', 'кубітів'])}`
export const layers = (n: number) => `${n} ${plural(n, ['шар', 'шари', 'шарів'])}`

export function modelName(m: ModelInfo): string {
  if (m.kind === 'classical') return 'Класична (лінійна на 1280 ознаках)'
  const base = `${ENCODING_UK[m.encoding ?? ''] ?? m.encoding}, ${qubits(m.n ?? 0)}`
  return m.custom ? `${base}, γ = ${String(m.gamma).replace('.', ',')}, ${layers(m.layers)}` : base
}
