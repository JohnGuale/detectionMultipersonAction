async function loadProperties() {
  try {
    // Carga el archivo desde static
    const res = await fetch('/static/properties.json', { cache: 'no-store' });
    const props = await res.json();

    const setText = (key, value) => {
      const el = document.getElementById(key);
      if (el) el.textContent = value;
    };

    // Asignaciones principales
    setText('appTitle', props.appTitle);
    setText('header.title', props.header?.title);
    setText('header.subtitle', props.header?.subtitle);

    // Labels
    Object.entries(props.labels || {}).forEach(([k, v]) => {
      setText(`labels.${k}`, v);
    });

  } catch (err) {
    console.error('Error cargando properties.json:', err);
  }
}

document.addEventListener('DOMContentLoaded', loadProperties);
