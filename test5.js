import {AutoProcessor, RawImage, AutoModel} from 'https://cdn.jsdelivr.net/npm/@xenova/transformers';

// Load model and processor
const model = await AutoModel.from_pretrained('Xenova/modnet-onnx', { quantized: false });
const processor = await AutoProcessor.from_pretrained('Xenova/modnet-onnx');

// Load image from URL
const url = 'https://images.pexels.com/photos/5965592/pexels-photo-5965592.jpeg?auto=compress&cs=tinysrgb&w=1024';
const image = await RawImage.fromURL(url);

// Process image
const {pixel_values: input} = await processor(image);

// Predict alpha matte
const {output} = await model({input});
console.log('image', image)

// Convert output tensor to RawImage
const matteImage = await RawImage.fromTensor(output[0].mul(255).to('uint8')).resize(image.width, image.height);

console.log('matteImage', matteImage, output)

async function renderRawImage(image) {
    let rawCanvas = await image.toCanvas();
    const canvas = document.createElement('canvas');
    document.body.appendChild(canvas); // 将新创建的 Canvas 添加到页面中
    canvas.width = image.width;
    canvas.height = image.height;
    
    const ctx = canvas.getContext('2d');
    
    ctx.drawImage(rawCanvas, 0, 0);
    
}

renderRawImage(matteImage)

async function getForeground(rawImage, maskImage) {
    const rawCanvas = rawImage.toCanvas();
    const rawCtx = rawCanvas.getContext('2d');
  
    const maskCanvas = maskImage.toCanvas();
    const maskCtx = maskCanvas.getContext('2d');
  
    const rawImageData = rawCtx.getImageData(0, 0, rawCanvas.width, rawCanvas.height);
    const maskImageData = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
  
    for (let i = 0; i < rawImageData.data.length; i += 4) {
        // 把灰度通道值（RGB 都一样，这里取 R），赋到原图的透明通道（每个像素的第 4 个值）
        rawImageData.data[i + 3] = maskImageData.data[i];
    }
  
    rawCtx.putImageData(rawImageData, 0, 0);
    return rawCanvas;
  }
    
let foregroundCanvas = await getForeground(image, matteImage);

// 使用示例：
console.log('debug', foregroundCanvas);
// 模拟异步操作，确保在完成操作后才继续执行
foregroundCanvas.convertToBlob()
  .then(function(blob) {
    // 创建图片
    let img = new Image();

    // 创建 blob URL 并设置为图片的 src
    img.src = URL.createObjectURL(blob); 

    // 将图片添加到 body 中或者其他 HTML 元素
    document.body.appendChild(img); 
})
.catch(function(error) {
    // 捕获和处理 blob 创建过程中可能出现的错误
    console.error("Blob creation error: ", error);
});
