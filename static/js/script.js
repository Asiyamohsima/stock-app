function fillSymbol(symbol) {
  const input = document.querySelector('input[name="nm"]');
  if (input) {
    input.value = symbol;
    input.focus();
  }
}
