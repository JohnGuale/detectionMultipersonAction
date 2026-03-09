var deteccionesTiempoReal = [];
var framesSkipToAnalyze = 3;
var selDeviceType = 'local';
var detecciones = [];
var source = null;
var visualMode = 'operativo';

document.addEventListener('DOMContentLoaded', function () {
    const uploadForm = document.getElementById('upload-form');
    const uploadBtn = document.getElementById('upload-btn');
    const reuploadBtn = document.getElementById('reupload-btn');
    const initInfo = document.getElementById('init-info');
    const imgProcessed = document.getElementById('img-processed-video');
    const skipframes_ddMenuButton = document.getElementById("dropdown1");
    const skipframes_ddItems = document.querySelectorAll('#dd1 .dropdown-item');
    const dimension_ddMenuButton = document.getElementById("dropdown2");
    const dimension_ddItems = document.querySelectorAll('#dd2 .dropdown-item');
    const device_ddMenuButton = document.getElementById("device-dd-btn");
    const device_ddItems = document.querySelectorAll('#device-dd-container .dropdown-item');
    const device_input = document.getElementById("device-url");
    const labelWarning = document.getElementById('label-warning');
    const viewResultsBtn = document.getElementById('view-results');
    const closeModalBtn = document.getElementById('close-btn');
    const fullVideo = document.getElementById('full-video');
    const modal = document.getElementById('myModal');
    const realtime_count = document.getElementById('realtime-results-count');
    const realtime_list = document.getElementById('realtime-list');
    const policeImage = document.getElementById('police-image');
    const irSubContainer = document.getElementById('ir-info-subcontainer');
    const foundResults = document.getElementById('found-results');
    const realtimeSpinner = document.getElementById('realtime-spinner');
    const validate_uri = document.getElementById('validate-uri-info');
    const timestampsList = document.getElementById('timestamps-list');

    // --- Submit del formulario ---
    uploadForm.addEventListener('submit', function (e) {
        e.preventDefault();
        if ((selDeviceType === 'remote' && validarURLMultimedia(device_input.value)) || selDeviceType === 'local') {
            goToDetection();
            
            skipframes_ddMenuButton.disabled = true;
            dimension_ddMenuButton.disabled = true;
            uploadBtn.disabled = true;
            device_ddMenuButton.disabled = true;
            device_input.disabled = true;
            
            irSubContainer.style.display = 'none';
            foundResults.style.display = 'block';
            showToast("Procesando dispositivo de captura", "info");
        } else {
            showToast('Conexión ingresada es errónea o está vacía', 'error');
        }
    });

    // --- Validación de URL remota ---
    device_input.addEventListener('keyup', _ => {
        if (!validarURLMultimedia(device_input.value)) {
            validate_uri.style.display = device_input.value !== '' ? 'block' : 'none';
        } else {
            validate_uri.style.display = 'none';
        }
    });

    // --- Iniciar detección en vivo ---
    function goToDetection() {
        let init = false;
        let urlParam = selDeviceType === 'remote' ? fixUrlForGet(device_input.value) : 'local'; 
        source = new EventSource(`/live-actions-remote/${urlParam}/${visualMode}`);

        source.onmessage = function (event) {
            if (event.data === "EOF") {
                source.close();
                realtimeSpinner.style.display = 'none';
                showToast("Detección en vivo culminada correctamente", "success");

                // 🔑 Reactivar controles para reprocesar
                reuploadBtn.style.display = 'inline-block';
                uploadBtn.disabled = false;
                device_input.disabled = false;

                return;
            }

            const data = JSON.parse(event.data);
            if (event.data && !init) {
                reuploadBtn.style.display = 'inline-block';
                imgProcessed.style.display = 'block';
                initInfo.style.display = 'none';
                init = true;
            }
            imgProcessed.src = 'data:image/jpeg;base64,' + data.frame;
            if (data.detections) updateRealTimeDetections(data.detections);
        };

        source.onerror = function () {
            showToast("Error en el stream de detección en vivo", "error");
            source.close();
        };
    }

    /*function goToDetection() {
        let init = false;
        //const source = new EventSource(`/live-actions`);
        let urlParam = selDeviceType === 'remote' ? fixUrlForGet(device_input.value) : 'local'; 
        const source = new EventSource(`/live-actions-remote/${urlParam}`);
        source.onmessage = function (event) {
            if (event.data === "EOF") {
                source.close();
                realtimeSpinner.style.display = 'none';
                showToast("Detección en vivo culminada correctamente", "success");
            } else {
                const data = JSON.parse(event.data);
                if (event.data && !init) {
                    reuploadBtn.style.display = 'inline-block';
                    imgProcessed.style.display = 'block';
                    initInfo.style.display = 'none';
                    init = true;
                }
                imgProcessed.src = 'data:image/jpeg;base64,' + data.frame;
                if (data.detections) updateRealTimeDetections(data.detections);
            }
        };
    }*/

    // --- Reiniciar proceso ---
    /*reuploadBtn.addEventListener('click', function () {
        location.reload();
    });*/

    // --- Reiniciar proceso (Reset Total) ---
    reuploadBtn.addEventListener('click', function (e) {
        e.preventDefault(); // 🔑 BLOQUEO ABSOLUTO: Evita que el formulario se reenvíe solo

        if (source) {
            source.close();
            source = null;
        }

        deteccionesTiempoReal = [];
        realtime_list.innerHTML = '';
        realtime_count.innerText = '0';
        
        imgProcessed.src = '';
        imgProcessed.style.display = 'none';   
        initInfo.style.display = 'block';

        foundResults.style.display = 'none';
        irSubContainer.style.display = 'block';
        policeImage.src = '/static/assets/police.jpg'; // Volver al policía neutral
        const descripcion = document.getElementById('labels.descripcionResultado');
        if (descripcion) descripcion.innerText = ''; 
        
        realtimeSpinner.style.display = 'none';

        skipframes_ddMenuButton.disabled = false;
        dimension_ddMenuButton.disabled = false;
        uploadBtn.disabled = false;
        device_ddMenuButton.disabled = false;

        if (selDeviceType === 'local') {
            device_input.disabled = true;
        } else {
            device_input.disabled = false;
        }

        showToast("Proceso reiniciado. Listo para nueva detección.", "success");
    });



    // --- Dropdown de frames ---
    skipframes_ddItems.forEach(item => {
        item.addEventListener('click', function (e) {
            e.preventDefault();
            let selectedText = this.textContent.trim();
            
            // Asignar el modo basado en la selección
            if (selectedText.includes('Analítico')) visualMode = 'analitico';
            else if (selectedText.includes('Debug')) visualMode = 'debug';
            else visualMode = 'operativo';

            // Actualizar el texto del botón
            if (selectedText.includes("defecto")) selectedText = selectedText.replace(' (Por defecto)', '');
            skipframes_ddMenuButton.innerHTML = `<i class="fa-solid fa-eye"></i> ${selectedText}&nbsp;`;
        });
    });

    // --- Dropdown de dimensión ---
    dimension_ddItems.forEach(item => {
        item.addEventListener('click', function (e) {
            e.preventDefault();
            let selectedText = this.textContent.trim();
            selDimension = selectedText;
            dimension_ddMenuButton.innerHTML = `<i class="fa-solid fa-male"></i> Estimación postural: ${selectedText}&nbsp;`;
        });
    });

    // --- Dropdown de dispositivo ---
    device_ddItems.forEach(item => {
        item.addEventListener('click', function (e) {
            e.preventDefault();
            let selectedText = this.textContent.trim();
            if (selectedText.includes('local')) {
                selDeviceType = 'local';
                device_input.disabled = true;
                device_input.placeholder = 'Se tomará el dispositivo local para el análisis';
            } else {
                selDeviceType = 'remote';
                device_input.disabled = false;
                device_input.placeholder = 'Ingrese la cadena de conexión al dispositivo remoto (HTTP - RTSP)';
            }
            device_ddMenuButton.innerText = selectedText;
        });
    });


    closeModalBtn.addEventListener('click', function () {
        closeModal();
    });

    document.addEventListener('keyup', function (event) {
        if (event.key === 'Escape' && modal.style.display === 'block') {
            closeModal();
        }
    });

    function closeModal() {
        modal.style.display = "none";
    }

    // --- Actualizar detecciones en vivo ---
function updateRealTimeDetections(detections) {
    console.log("Detecciones recibidas:", detections);

    // Filtrar solo acciones relevantes (excluir NEUTRAL)
    const accionesValidas = detections.filter(d => d === 'PELEAR' || d === 'DISTURBIO');

    // Acumular acciones detectadas en la sesión
    deteccionesTiempoReal = [...new Set([...deteccionesTiempoReal, ...accionesValidas])];
    realtime_count.innerText = deteccionesTiempoReal.length;

    // Mostrar cada acción recibida en este frame
    accionesValidas.forEach(behavior => {
        let labelHTML = '';
        if (behavior === 'PELEAR') {
            labelHTML = `<label><i class="fa-solid fa-fist-raised"></i> Pelea detectada</label>`;
        } else if (behavior === 'DISTURBIO') {
            labelHTML = `<label><i class="fa-solid fa-exclamation-triangle"></i> Disturbio detectado</label>`;
        }

        // Añadir solo si no existe ya en la lista
        if (labelHTML && !realtime_list.innerHTML.includes(labelHTML)) {
            realtime_list.innerHTML += `${realtime_list.innerText !== '' ? '<br>' : ''}${labelHTML}`;
        }
    });

    // Cambiar imagen si hay alguna detección
    if (deteccionesTiempoReal.length > 0 && !policeImage.src.includes('angry')) {
        policeImage.src = '/static/assets/angry_police.jpg';
    }

    // Actualizar texto principal con todas las acciones detectadas
    const descripcion = document.getElementById('labels.descripcionResultado');
    if (descripcion) {
        descripcion.innerText = deteccionesTiempoReal.join(' | ');
    }
}

    // --- Validar URL multimedia ---
    function validarURLMultimedia(url) {
        const regex = /^(rtsp|http|https):\/\/[^\s/$.?#].[^\s]*$/i;
        return regex.test(url);
    }

    // --- Fix URL para GET ---
    function fixUrlForGet(url) {
        return url ? url.replaceAll('/', '{slash}') : null;
    }
    // --- Mostrar resumen en el modal ---
// --- Mostrar resumen en el modal ---
function showLiveSummary() {
    const timestamp = new Date().getTime();

    fetch(`/static/videos/live/live_report.json?t=${timestamp}`)
        .then(response => response.json())
        .then(events => {
            const summaryList = document.getElementById('live-summary-list');
            summaryList.innerHTML = '';

            if (!events.length) {
                summaryList.innerHTML = '<p class="text-center">No se detectaron acciones.</p>';
            } else {
                events.forEach(ev => {
                    let label = '';
                    let colorBorde = '';

                    if (ev.tipo_evento === 'PELEAR') {
                        label = `<strong><i class="fa-solid fa-fist-raised" style="color: red;"></i> Pelea detectada</strong><br>
                                 <small>Duración: ${ev.duracion_total}s | Hora: ${ev.hora_inicio} - ${ev.hora_fin} | Día: ${ev.fecha_inicio}</small>`;
                        colorBorde = 'red';
                    } else if (ev.tipo_evento === 'DISTURBIO') {
                        label = `<strong><i class="fa-solid fa-exclamation-triangle" style="color: orange;"></i> Disturbio detectado</strong><br>
                                 <small>Duración: ${ev.duracion_total}s | Hora: ${ev.hora_inicio} - ${ev.hora_fin} | Día: ${ev.fecha_inicio}</small>`;
                        colorBorde = 'orange';
                    }
                    let imagenHTML = '';
                    if (ev.ruta_imagen) {
                        imagenHTML = `<div style="text-align: center; margin-top: 10px;">
                                        <img src="${ev.ruta_imagen}?t=${timestamp}" alt="Captura del evento" 
                                             style="max-width: 100%; max-height: 250px; border: 2px solid ${colorBorde}; border-radius: 5px;">
                                      </div>`;
                    }

                    summaryList.innerHTML += `<div class="list-group-item">${label}${imagenHTML}</div>`;
                });
            }

            document.getElementById('myModal').style.display = "block";
        });
}

// --- Vincular botón "Visualizar Resumen" ---
viewResultsBtn.addEventListener('click', function () {
    if (deteccionesTiempoReal.length) {
        showLiveSummary(); // para detección en vivo
    } else {
        showToast("No hay resultados para mostrar", "info");
    }
});

});
