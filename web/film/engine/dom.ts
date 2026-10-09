type Child = Node | string | null | undefined | false;

export function h<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className = "",
  ...children: Child[]
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className) node.className = className;
  for (const child of children) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child);
  }
  return node;
}

export function svgUse(symbol: string, className = ""): SVGSVGElement {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  if (className) svg.setAttribute("class", className);
  const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
  use.setAttribute("href", `#${symbol}`);
  svg.append(use);
  return svg;
}

export function icon(markup: string, className = ""): HTMLSpanElement {
  const holder = h("span", className);
  holder.innerHTML = markup;
  return holder;
}
