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
    // const canvas = document.getElementById('canvas');
    canvas.width = image.width;
    canvas.height = image.height;
    
    const ctx = canvas.getContext('2d');
    
    ctx.drawImage(rawCanvas, 0, 0);
    
}

renderRawImage(matteImage)

// async function createForegroundViaCanvas(rawImage, maskImage) {
//     // 首先，获取原图和遮罩的图像数据
//     const rawImageData = rawImage.toCanvas().getContext('2d').getImageData(0, 0, rawImage.width, rawImage.height);
//     const maskImageData = maskImage.toCanvas().getContext('2d').getImageData(0, 0, maskImage.width, maskImage.height);
  
//     // 遍历像素数据
//     for (let i = 0; i < rawImageData.data.length; i += 4) {
//       // 检查相应的遮罩数据，3rd index为alpha channel
//       if (maskImageData.data[i + 3] !== 255) {
//         // 如果在遮罩的这个像素点不为255（不是前景），则将原图在这个点的alpha设为0（透明）
//         rawImageData.data[i + 3] = 0;
//       }
//     }
  
//     // 现在你的 rawImageData 仅包含具有原色的前景像素，其余像素全透明
//     // 创建一个新的canvas，将处理后的图像绘制到新的canvas中
//     const canvas = document.createElement('canvas');
//     canvas.width = rawImage.width;
//     canvas.height = rawImage.height;
//     const context = canvas.getContext('2d');
//     context.putImageData(rawImageData, 0, 0);
    
//     document.body.appendChild(canvas); // 将新创建的 Canvas 添加到页面中
//     // 现在你的 canvas 就包含了前景图像
//     return canvas;
//   }

async function getForeground(rawImage, maskImage) {
    const rawCanvas = rawImage.toCanvas();
    const rawCtx = rawCanvas.getContext('2d');
  
    const maskCanvas = maskImage.toCanvas();
    const maskCtx = maskCanvas.getContext('2d');
  
    const rawImageData = rawCtx.getImageData(0, 0, rawCanvas.width, rawCanvas.height);
    const maskImageData = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
  
    for (let i = 0; i < rawImageData.data.length; i += 4) {
        // 获取遮罩图像的灰度值
        // let grayscale = maskImageData.data[i];
  
        // 如果此像素的灰度值为0（即背景），就把原图对应位置的像素设为透明
        // console.log('grayscale', grayscale);
        // if (grayscale <= 128) {
        //     rawImageData.data[i + 3] = 0;
        // }
        // 把灰度通道值（RGB 都一样，这里取 R），赋到原图的透明通道（每个像素的第 4 个值）
        rawImageData.data[i + 3] = maskImageData.data[i];
    }
  
    rawCtx.putImageData(rawImageData, 0, 0);
    // document.body.appendChild(rawCanvas); // 将新创建的 Canvas 添加到页面中
    return rawCanvas;
  }
  
//   let foregroundCanvas = await getForeground(rawImage, maskImage);
  
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

// const rawImage = {
//     data: new Uint8ClampedArray(3 * 100 * 100), // 假设图像尺寸为 100x100
//     width: 100,
//     height: 100,
//     channels: 3
//   };
// renderRawImage(rawImage)

// renderRawImage(matteImage)
// // Create foreground image
// const canvas = document.createElement('canvas');
// const ctx = canvas.getContext('2d');
// // Create two new canvas contexts
// const ctx1 = document.createElement("canvas").getContext("2d");
// const ctx2 = document.createElement("canvas").getContext("2d");

// // Draw the image and matteImage onto the two canvas respectively
// ctx1.drawImage(input, 0, 0);
// ctx2.drawImage(matteImage, 0, 0);

// canvas.width = image.width;
// canvas.height = image.height;

// const imageData = ctx.getImageData(0, 0, image.width, image.height);
// const matteData = ctx.getImageData(0, 0, matteImage.width, matteImage.height);

// // Apply matte to foreground
// for (let i = 0; i < imageData.data.length; i += 4) {
//   const matteVal = matteData.data[i] / 255;
//   imageData.data[i] = imageData.data[i] * matteVal;  // R
//   imageData.data[i + 1] = imageData.data[i + 1] * matteVal;  // G
//   imageData.data[i + 2] = imageData.data[i + 2] * matteVal;  // B
// }

// // Save foreground image
// ctx.putImageData(matteImage, 0, 0);
// // Now foregroundImage contains the foreground image you can save or display it as needed

// // Rest of the code above...

// // Convert canvas to dataURL
// const foregroundDataUrl = canvas.toDataURL();

// // Render the output on HTML
// document.getElementById('output-image').src = foregroundDataUrl;
// console.log('finish')

// function generateForeground(originalImage, mask, width, height) {
//     let foreground = new Uint8ClampedArray(originalImage.length);
//     const channels = 4; // RGB and alpha
  
//     for (let i = 0; i < width * height * channels; i += channels) {
//       if (mask[i] === 0) { // Assuming the mask is grayscale
//         // If the mask is black, the foreground pixel is transparent
//         foreground[i] = 0;     // R
//         foreground[i + 1] = 0; // G
//         foreground[i + 2] = 0; // B
//         foreground[i + 3] = 0; // A
//       } else {
//         // If the mask is white, copy the original pixel
//         foreground[i] = originalImage[i];     // R
//         foreground[i + 1] = originalImage[i + 1]; // G
//         foreground[i + 2] = originalImage[i + 2]; // B
//         foreground[i + 3] = 255; // A
//       }
//     }
  
//     return foreground;
//   }

// let foreground = generateForeground(image, matteImage, image.width, image.height);
// console.log('done', foreground);
//   outputImage.save('outputx.png');

// Assume you have a canvas and context
// let canvas = document.createElement('canvas');
// document.body.appendChild(canvas);
// canvas.width = image.width;
// canvas.height = image.height;
// // let ctx = canvas.getContext('2d');

// function extractForeground(originalImg, maskImg) {
//     if (
//         originalImg.width !== maskImg.width ||
//         originalImg.height !== maskImg.height
//     ) {
//         throw new Error('原图像和遮罩图像尺寸不匹配');
//     }

//     // 创建一个新的 Uint8ClampedArray，用于存储前景图像的数据
//     const foregroundData = new Uint8ClampedArray(originalImg.data.length);

//     for (let i = 0; i < originalImg.data.length; i += 4) {
//         // 判断遮罩图像的 pixel 是否为白色（即 RGB 的值都大等于240）
//         if (
//             maskImg.data[i] >= 240 &&     // Red
//             maskImg.data[i + 1] >= 240 && // Green
//             maskImg.data[i + 2] >= 240    // Blue
//         ) {
//             // 如果是白色，则从原图像中提取对应 pixel 的数据
//             foregroundData[i] = originalImg.data[i];       // Red
//             foregroundData[i + 1] = originalImg.data[i + 1]; // Green
//             foregroundData[i + 2] = originalImg.data[i + 2]; // Blue
//             foregroundData[i + 3] = originalImg.data[i + 3]; // Alpha
//         } else {
//             // 如果不是白色，则设为透明
//             foregroundData[i] = 0;     // Red
//             foregroundData[i + 1] = 0; // Green
//             foregroundData[i + 2] = 0; // Blue
//             foregroundData[i + 3] = 0; // Alpha
//         }
//     }

//     return new ImageData(foregroundData, originalImg.width, originalImg.height);
// }

// let fore = extractForeground(image, matteImage);
// renderImageToCanvas(fore, canvas);
// function renderImageToCanvas(imageData, canvas) {
//     const ctx = canvas.getContext('2d');
//     ctx.putImageData(imageData, 0, 0);
// }
