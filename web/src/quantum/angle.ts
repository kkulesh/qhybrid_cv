// Safe parser for angle expressions such as "pi/2", "-0.5*pi", "3*pi/4", "1.2".
// Grammar: expr = term (('+'|'-') term)*; term = factor (('*'|'/') factor)*;
//          factor = ('+'|'-') factor | number | 'pi' | 'π' | '(' expr ')'

export function parseAngle(src: string): number | null {
  const s = src.replace(/\s+/g, '').toLowerCase()
  let i = 0
  const peek = () => s[i]
  function expr(): number {
    let v = term()
    while (peek() === '+' || peek() === '-') {
      const op = s[i++]
      const r = term()
      v = op === '+' ? v + r : v - r
    }
    return v
  }
  function term(): number {
    let v = factor()
    while (peek() === '*' || peek() === '/') {
      const op = s[i++]
      const r = factor()
      v = op === '*' ? v * r : v / r
    }
    return v
  }
  function factor(): number {
    if (peek() === '+') { i++; return factor() }
    if (peek() === '-') { i++; return -factor() }
    if (peek() === '(') {
      i++
      const v = expr()
      if (s[i++] !== ')') throw new Error('missing )')
      return v
    }
    if (s.startsWith('pi', i)) { i += 2; return Math.PI }
    if (peek() === 'π') { i++; return Math.PI }
    const m = /^\d*\.?\d+(e[+-]?\d+)?/.exec(s.slice(i))
    if (!m) throw new Error('number expected')
    i += m[0].length
    return parseFloat(m[0])
  }
  try {
    if (!s) return null
    const v = expr()
    return i === s.length && Number.isFinite(v) ? v : null
  } catch {
    return null
  }
}

/** Human-friendly form: multiples of pi/8 are shown as fractions of π. */
export function formatAngle(v: number, digits = 3): string {
  const k = (v / Math.PI) * 8
  if (Math.abs(k - Math.round(k)) < 1e-9) {
    const n = Math.round(k)
    if (n === 0) return '0'
    const g = gcd(Math.abs(n), 8)
    const num = n / g
    const den = 8 / g
    const head = num === 1 ? '' : num === -1 ? '-' : `${num}`
    return den === 1 ? `${head}π` : `${head}π/${den}`
  }
  return v.toFixed(digits).replace(/\.?0+$/, '')
}

function gcd(a: number, b: number): number {
  return b === 0 ? a : gcd(b, a % b)
}
