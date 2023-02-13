import {WasmWrapper} from './wasm.js';

let wasmWrapper = new WasmWrapper();

export function start() {
    loadTFLiteModel('./model/linear.tflite');
}

function loadTFLiteModel(modelpath) {
    fetch(modelpath)
    .then(response => checkStatus(response) && response.arrayBuffer())
    .then(buffer => {
        console.log(buffer);
        let bytes = new Uint8Array( buffer );
        let len = bytes.byteLength;
        console.log(bytes);
        console.log(len);
        wasmWrapper.test_linear(bytes, len, 2.0);

    })
    .catch(err => console.error(err)); // Never forget the final catch!
}

function checkStatus(response) {
    if (!response.ok) {
        throw new Error(`HTTP ${response.status} - ${response.statusText}`);
    }
    return response;
}

