import { simd, threads } from "https://unpkg.com/wasm-feature-detect?module";

export class WasmWrapper {
    constructor() {
        this.loaded = false;
        this.angle = 0;
        this.checkFeatures_().then(({useSimd, useThread}) => {
            if (!useThread) {
                console.warn("Threads disabled, seems that the security requirements for SharedArrayBuffer are not met")
                return;
            }
            let dir = useSimd? "simd" : "nonsimd";
            this.loadModuleScript_("./linear2.js").then(() => {
                createModule().then(instance => {
                    this.wasmModule = instance;
                    this.loaded = true;
                });
            });
        })
    }

    /** @private */
    async checkFeatures_() {
        let useSimd = await simd();
        let useThread = await threads();
        console.log(useSimd, useThread);
        return {useSimd, useThread};
    }

    /** @private */
    loadModuleScript_(jsUrl) {
        return new Promise((resolve, reject) => {
            let script = document.createElement('script');
            script.onload = (() => {
                resolve();
            });
            script.onerror = (() => {
                reject();
            });
            script.src = jsUrl;
            document.body.appendChild(script);
        });
    }

    /** @private */
    createBuffer_(buffer_len) {
        return this.wasmModule._malloc(buffer_len);
    }

    /** @private */
    freeBuffer_(buffer) {
        this.wasmModule._free(buffer);
    }

    test_linear(modelData, dataLen, n) {
        const buffer = this.createBuffer_(dataLen);
        this.wasmModule.HEAPU8.set(modelData, buffer);
        this.result = this.wasmModule.ccall(
            'test_linear',
            'number',
            ['number', 'number', 'number'],
            [buffer, dataLen, n]);
            console.log('test linear result = ' + this.result);
        this.freeBuffer_(buffer);
    }
}