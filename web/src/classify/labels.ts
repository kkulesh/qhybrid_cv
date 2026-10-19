import type { ModelInfo } from './api'

export const CLASS_UK: Record<string, string> = {
  aeroplane: 'літак', bicycle: 'велосипед', bird: 'птах', boat: 'човен', bottle: 'пляшка',
  bus: 'автобус', car: 'автомобіль', cat: 'кіт', chair: 'стілець', cow: 'корова',
  diningtable: 'обідній стіл', dog: 'собака', horse: 'кінь', motorbike: 'мотоцикл', person: 'людина',
  pottedplant: 'рослина в горщику', sheep: 'вівця', sofa: 'диван', train: 'потяг', tvmonitor: 'монітор',
}

export const uk = (label: string) => CLASS_UK[label] ?? label

export const ENCODING_UK: Record<string, string> = { amplitude: 'Амплітудне', angle: 'Кутове', zz: 'ZZ-відображення' }

export function modelName(m: ModelInfo): string {
  if (m.kind === 'classical') return 'Класична (лінійна на 1280 ознаках)'
  return `${ENCODING_UK[m.encoding ?? ''] ?? m.encoding}, ${m.n} кубітів`
}
