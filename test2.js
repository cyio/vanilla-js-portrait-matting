import { AutoProcessor, VitMatteForImageMatting, RawImage } from '@xenova/transformers';

// Load processor and model
const processor = await AutoProcessor.from_pretrained('Xenova/vitmatte-small-composition-1k');
const model = await VitMatteForImageMatting.from_pretrained('Xenova/vitmatte-small-composition-1k');

// Load image and trimap
const image = await RawImage.fromURL('https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/vitmatte_image.png');
const trimap = await RawImage.fromURL('https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/vitmatte_trimap.png');

// Prepare image + trimap for the model
const inputs = await processor(image, trimap);

// Predict alpha matte
const { alphas } = await model(inputs);
// Tensor {
//   dims: [ 1, 1, 640, 960 ],
//   type: 'float32',
//   size: 614400,
//   data: Float32Array(614400) [ 0.9894027709960938, 0.9970508813858032, ... ]
// }
