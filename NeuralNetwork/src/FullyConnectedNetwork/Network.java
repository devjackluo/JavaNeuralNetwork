package FullyConnectedNetwork;

import TrainSet.TrainSet;
import parser.Attribute;
import parser.Node;
import parser.Parser;
import parser.ParserTools;


import java.io.*;
import java.util.Arrays;

import static FullyConnectedNetwork.NetworkTools.createRandomArray;
import static FullyConnectedNetwork.NetworkTools.createRandomMultiArray;

public class Network implements Serializable{

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
            this.bias[i] = createRandomArray(NETWORK_LAYER_SIZES[i], -1.0, 1.0);


            this.error_signal[i] = new double[NETWORK_LAYER_SIZES[i]];
            this.output_derivatives[i] = new double[NETWORK_LAYER_SIZES[i]];



            //the first layer doesn't have a weight because no previous layer
            if(i > 0){
                //this.weights[i] = new double[NETWORK_LAYER_SIZES[i]][NETWORK_LAYER_SIZES[i-1]];
                this.weights[i] = createRandomMultiArray(NETWORK_LAYER_SIZES[i],NETWORK_LAYER_SIZES[i-1], -1.0, 1.0);
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

    public void trainSet(TrainSet trainSet, int loop, int batch_size){
        if(trainSet.INPUT_SIZE != INPUT_SIZE || trainSet.OUTPUT_SIZE != OUTPUT_SIZE){
            return;
        }
        for(int i = 0 ; i < loop; i++){
            TrainSet batch = trainSet.extractBatch(batch_size);
            for(int b = 0; b < batch.size(); b++){
                this.train(batch.getInput(b), batch.getOutput(b), 0.3);
            }
        }
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

    /*
    public void saveNetwork(String file) throws Exception{
        File f = new File(file);
        ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(f));
        out.writeObject(this);
        out.flush();
        out.close();
    }

    public static Network loadNetwork(String file) throws Exception{

        File f = new File(file);
        ObjectInputStream out = new ObjectInputStream(new FileInputStream(f));
        Network net = (Network)out.readObject();
        out.close();
        return net;
    }
    */


    public void saveNetwork(String fileName) throws Exception {
        Parser p = new Parser();
        p.create(fileName);
        Node root = p.getContent();
        Node netw = new Node("Network");
        Node ly = new Node("Layers");
        netw.addAttribute(new Attribute("sizes", Arrays.toString(this.NETWORK_LAYER_SIZES)));
        netw.addChild(ly);
        root.addChild(netw);
        for (int layer = 1; layer < this.NETWORK_SIZE; layer++) {

            Node c = new Node("" + layer);
            ly.addChild(c);
            Node w = new Node("weights");
            Node b = new Node("biases");
            c.addChild(w);
            c.addChild(b);

            b.addAttribute("values", Arrays.toString(this.bias[layer]));

            for (int we = 0; we < this.weights[layer].length; we++) {

                w.addAttribute("" + we, Arrays.toString(weights[layer][we]));
            }
        }
        p.close();
    }

    public static Network loadNetwork(String fileName) throws Exception {

        Parser p = new Parser();

        p.load(fileName);
        String sizes = p.getValue(new String[] { "Network" }, "sizes");
        int[] si = ParserTools.parseIntArray(sizes);
        Network ne = new Network(si);

        for (int i = 1; i < ne.NETWORK_SIZE; i++) {
            String biases = p.getValue(new String[] { "Network", "Layers", new String(i + ""), "biases" }, "values");
            double[] bias = ParserTools.parseDoubleArray(biases);
            ne.bias[i] = bias;

            for(int n = 0; n < ne.NETWORK_LAYER_SIZES[i]; n++){

                String current = p.getValue(new String[] { "Network", "Layers", new String(i + ""), "weights" }, ""+n);
                double[] val = ParserTools.parseDoubleArray(current);

                ne.weights[i][n] = val;
            }
        }
        p.close();
        return ne;

    }

    public static void main(String[] args){

        Network net = new Network(40,35,30,35,3);


//        double[] input = new double[]{0.1,0.5,0.2,0.8};
//        double[] target = new double[]{0.5,0.0,1,0.0};
//
//        double[] input2 = new double[]{0.8,0.2,0.5,0.8};
//        double[] target2 = new double[]{0.0,1,1,0.0};
//
//
//
//        for(int i = 0; i < 100000; i++){
//            net.train(input, target, 0.2);
//            net.train(input2, target2, 0.2);
//
//        }
//
//        double[] results = net.calculate(input);
//        System.out.println(Arrays.toString(results));
//
//        results = net.calculate(input2);
//        System.out.println(Arrays.toString(results));



        TrainSet set = new TrainSet(40, 3);
        set.addSampleData(10);
//        set.addData(new double[]{0.1,0.2,0.3,0.4}, new double[]{0.9,0.1,0.9,0.1});
//        set.addData(new double[]{0.9,0.8,0.7,0.6}, new double[]{0.1,0.9,0.1,0.9});
//        set.addData(new double[]{0.3,0.8,0.1,0.4}, new double[]{0.3,0.7,0.3,0.7});
//        set.addData(new double[]{0.9,0.8,0.1,0.2}, new double[]{0.7,0.3,0.7,0.3});


        net.trainSet(set, 100000, set.size()/3);

        for(int i = 0; i < set.size(); i++){
            //System.out.println(Arrays.toString(net.calculate(set.getInput(i))));
            System.out.println("("+ Arrays.toString(set.getOutput(i)) + ") :" + Arrays.toString(net.calculate(set.getInput(i))));
        }




    }

}
