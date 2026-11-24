import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import './styles/global.css';
// Тема «Инженерная редакция»: инженерный синий, миллиметровка, крупная типографика.
// Идёт после базовых стилей и перекрывает токены и формы.
import './styles/theme.css';

const container = document.getElementById('root');
if (!container) throw new Error('Не найден корневой элемент #root');

// Автозапуск проверки по ?q= — удобно для скриншотов и для ссылок из чата.
const params = new URLSearchParams(window.location.search);
const preset = params.get('q');
if (preset) {
  try {
    sessionStorage.setItem('preset_text', preset);
  } catch {
    /* приватный режим — проверка просто не запустится автоматически */
  }
}

createRoot(container).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
