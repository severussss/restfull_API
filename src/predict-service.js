import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';
import tf from '@tensorflow/tfjs-node';
import metadata from '../model/metadata.json' with { type: 'json' };
 
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
 
export async function predictImage(photo) {
  const modelPath = `file://${resolve(__dirname, '..', 'model', 'model.json')}`;
  const model = await tf.loadLayersModel(modelPath);
 
  const tensor = tf.node
    .decodeImage(photo)
    .resizeNearestNeighbor([224, 224])
    .expandDims()
    .toFloat();
 
  const predict = model.predict(tensor);
  const score = await predict.data();
  const confidenceScore = Math.max(...score);
  const label = tf.argMax(predict, 1).dataSync()[0];
 
  const diseaseLabels = metadata.labels;
  const diseaseLabel = diseaseLabels[label];
 
  return { confidenceScore, diseaseLabel };
}