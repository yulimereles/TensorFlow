// Elementos del DOM
const buttonLearn = document.querySelector('#aprender');
const isTrainingSpan = document.querySelector('#is-training');
const trainingCompleteSpan = document.getElementById('text-train-success');
const buttonPredict = document.querySelector('#predecir');
const inputField = document.querySelector('#input');
const outputText = document.querySelector('#output-text');

// Constantes
const ARRAY_LENGTH = 9;

// Modelo
const model = tf.sequential();

// Capas
model.add(tf.layers.dense({
    units: 1,
    inputShape: [1]
}
));

// Compilación
model.compile(
    {
        loss: 'meanSquaredError',
        optimizer: 'sgd'
    });
    
// Eventos para entrenar
buttonLearn.addEventListener('click', async () => {

    isTrainingSpan.classList.remove('visually-hidden');

    const INPUT = Array.from({ length: ARRAY_LENGTH }, (x, i) => i - 6);

    const OUTPUT = INPUT.map(x => 2 * x + 6);
    console.log(INPUT, OUTPUT);

    const xs = tf.tensor2d(INPUT, [INPUT.length, 1]);
    const ys = tf.tensor2d(OUTPUT, [OUTPUT.length, 1]);

    await model.fit(xs, ys, { epochs: 1000 });


    isTrainingSpan.classList.add('visually-hidden');
    trainingCompleteSpan.classList.remove('visually-hidden');
    buttonLearn.disabled = true;
    document.getElementById('predecir-section').classList.remove('visually-hidden');

})

// Evento para predecir
buttonPredict.addEventListener('click', async () => {
    const output = model.predict(tf.tensor2d([+inputField.value], [1, 1]));
    outputText.innerText = `El resultado de la predicción para ${inputField.value} es: ${output.dataSync()[0]}`;
})