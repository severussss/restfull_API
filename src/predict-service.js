import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';
import tf from '@tensorflow/tfjs-node';
import metadata from '../model/metadata.json' with { type: 'json' };

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

export async function predictImage(photo) {
  let tensor = tf.node.decodeImage(photo);

  if (tensor.shape[2] === 4) {
    tensor = tensor.slice([0, 0, 0], [-1, -1, 3]);
  }

  tensor = tensor.resizeNearestNeighbor([224, 224]).expandDims().toFloat();

  const modelPath = `file://${resolve(__dirname, '..', 'model', 'model.json')}`;
  const model = await tf.loadLayersModel(modelPath);

  const predict = model.predict(tensor);
  const score = await predict.data();
  const confidenceScore = Math.max(...score);
  const label = tf.argMax(predict, 0.1).dataSync()[0];

  const diseaseLabels = metadata.labels;
  const diseaseLabel = diseaseLabels[label];

  return { confidenceScore, diseaseLabel };
}