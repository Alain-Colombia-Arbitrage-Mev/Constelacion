document.addEventListener('DOMContentLoaded', () => {
    fetch('http://localhost:3000/predictions')
        .then(response => response.json())
        .then(data => {
            console.log(data);
            const tableBody = document.querySelector('#signalsTable tbody');
            tableBody.innerHTML = ''; // Limpiar la tabla antes de agregar nuevas filas

            // Iterar sobre cada predicción en los datos recibidos
            data.forEach((prediction, index) => {

                console.log("pre", prediction);

                const token = ['prediction', 'prediction-avax'][index]; // Asume que los archivos están en el mismo orden

                // Crear una nueva fila para cada señal recibida
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${prediction.token}</td>
                    <td>$${prediction.predicted_price}</td>
                    <td>${new Date(prediction.prediction_date).toLocaleDateString()}</td>
                    <td>${prediction.prediction_days}</td>
                `;
                tableBody.appendChild(row);
            });
        })
        .catch(error => console.error('Error al obtener las señales:', error));
});
