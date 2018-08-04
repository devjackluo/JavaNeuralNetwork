package FullyConnectedNetwork;

import java.util.Arrays;

import static FullyConnectedNetwork.NetworkTools.createRandomArray;
import static FullyConnectedNetwork.NetworkTools.createRandomMultiArray;

public class Network {

    private double[][] output;
    private double[][][] weights;
    private double[][] bias;

    private double[][] error_signal;
    private double[][] output_derivatives;


    public final int[] NETWORK_LAYER_SIZES;
    public final int INPUT_SIZE;
    public final int OUTPUT_SIZE;
    public final int NETWORK_SIZE;

    public Network(int... Network_Layer_Size){
        this.NETWORK_LAYER_SIZES = Network_Layer_Size;
        this.NETWORK_SIZE = NETWORK_LAYER_SIZES.length;
        this.INPUT_SIZE = NETWORK_LAYER_SIZES[0];
        this.OUTPUT_SIZE = NETWORK_LAYER_SIZES[NETWORK_SIZE-1];

        //first dimension keeps track of which layer it is on
        //second dimension keeps track of which node of the layer
        this.output = new double[NETWORK_SIZE][];
        this.bias = new double[NETWORK_SIZE][];
        //weights get a special third dimension to keep track of node from previous layer
        //where 2nd dimension is previous and 3rd is current node (previous --weight--> current) but doesn't matter
        // currently grabbing weights (current <--weight-- previous)
        this.weights = new double[NETWORK_SIZE][][];

        this.error_signal = new double[NETWORK_SIZE][];
        this.output_derivatives = new double[NETWORK_SIZE][];


        for(int i = 0; i < NETWORK_SIZE; i++){

            this.output[i] = new double[NETWORK_LAYER_SIZES[i]];
            //this.bias[i] = new double[NETWORK_LAYER_SIZES[i]];
            this.bias[i] = createRandomArray(NETWORK_LAYER_SIZES[i], 0.3, 0.7);


            this.error_signal[i] = new double[NETWORK_LAYER_SIZES[i]];
            this.output_derivatives[i] = new double[NETWORK_LAYER_SIZES[i]];



            //the first layer doesn't have a weight because no previous layer
            if(i > 0){
                //this.weights[i] = new double[NETWORK_LAYER_SIZES[i]][NETWORK_LAYER_SIZES[i-1]];
                this.weights[i] = createRandomMultiArray(NETWORK_LAYER_SIZES[i],NETWORK_LAYER_SIZES[i-1], -0.3, 0.5);
            }
        }

    }

    public double[] calculate(double... input){
        if(input.length != this.INPUT_SIZE){
            return null;
        }

        //set the first layer to be the input we gave to network
        this.output[0] = input;

        //for all hidden layers
        for(int layer = 1; layer < NETWORK_SIZE; layer++){
            //for all neurons(nodes) in those layers
            for(int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron++){

                //could put this after calculating weights but then bias won't have much impact
                double sum = bias[layer][neuron];

                //take the value every previous neuron and multiple it by the weight going to towards this neuron
                //add it to sum
                for(int prevNeuron = 0; prevNeuron < NETWORK_LAYER_SIZES[layer-1]; prevNeuron++){
                    sum += output[layer-1][prevNeuron] * weights[layer][neuron][prevNeuron];
                }

                //sigmoid the sum (0.0000- to 1.0000-)
                output[layer][neuron] = sigmoid(sum);

                output_derivatives[layer][neuron] = output[layer][neuron] * (1 - output[layer][neuron]);

            }
        }

        //return the last layer (the predicted value / guess)
        return output[NETWORK_SIZE-1];
    }


    public void train(double[] input, double[] target, double eta){
        if(input.length != INPUT_SIZE || target.length != OUTPUT_SIZE){
            return;
        }
        calculate(input);
        backpropError(target);
        updateWeights(eta);
    }

    public void backpropError(double[] target){
        for(int neuron = 0; neuron < NETWORK_LAYER_SIZES[NETWORK_SIZE-1]; neuron++){
            error_signal[NETWORK_SIZE-1][neuron] =
                    (output[NETWORK_SIZE-1][neuron] - target[neuron]) * output_derivatives[NETWORK_SIZE-1][neuron];
        }
        for(int layer = NETWORK_SIZE-2; layer > 0 ; layer--){
            for(int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron++) {
                double sum = 0;
                for(int nextNeuron = 0; nextNeuron < NETWORK_LAYER_SIZES[layer+1]; nextNeuron++){
                    sum += weights[layer+1][nextNeuron][neuron] * error_signal[layer + 1][nextNeuron];
                }
                this.error_signal[layer][neuron] = sum * output_derivatives[layer][neuron];
            }
        }
    }

    public void updateWeights(double eta){
        for(int layer = 1; layer < NETWORK_SIZE ; layer++){
            for(int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron++) {

                double delta = - eta * error_signal[layer][neuron];
                bias[layer][neuron] += delta;

                for(int prevNeuron = 0; prevNeuron < NETWORK_LAYER_SIZES[layer-1]; prevNeuron++){
                    //double deltaWeight = - eta * output[layer-1][prevNeuron] * error_signal[layer][neuron];
                    double deltaWeight = delta * output[layer-1][prevNeuron];
                    this.weights[layer][neuron][prevNeuron] += deltaWeight;
                }

//                double delta = - eta * error_signal[layer][neuron];
//                bias[layer][neuron] += delta;
            }
        }
    }


    private double sigmoid(double x){
        return (1d / (1 + Math.exp(-x)));
    }

    public static void main(String[] args){
        Network net = new Network(4,3,2,3,4);

        double[] input = new double[]{0.1,0.5,0.2,0.8};
        double[] target = new double[]{0.5,0.0,1,0.0};

        for(int i = 0; i < 100000; i++){
            net.train(input, target, 0.7);
        }

        double[] results = net.calculate(input);

        System.out.println(Arrays.toString(results));
        System.out.println("Pause");
    }

}
