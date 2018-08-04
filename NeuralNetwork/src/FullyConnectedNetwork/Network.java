package FullyConnectedNetwork;

public class Network {

    private double[][] output;
    private double[][][] weights;
    private double[][] bias;

    public final int[] NETWORK_LAYER_SIZE;
    public final int INPUT_SIZE;
    public final int OUTPUT_SIZE;
    public final int NETWORK_SIZE;

    public Network(int... Network_Layer_Size){
        this.NETWORK_LAYER_SIZE = Network_Layer_Size;
        this.NETWORK_SIZE = NETWORK_LAYER_SIZE.length;
        this.INPUT_SIZE = NETWORK_LAYER_SIZE[0];
        this.OUTPUT_SIZE = NETWORK_LAYER_SIZE[NETWORK_SIZE-1];

        //first dimension keeps track of which layer it is on
        //second dimension keeps track of which node of the layer
        this.output = new double[NETWORK_SIZE][];
        this.bias = new double[NETWORK_SIZE][];
        //weights get a special third dimension to keep track of node from previous layer
        this.weights = new double[NETWORK_SIZE][][];

        for(int i = 0; i < NETWORK_SIZE; i++){
            this.output[i] = new double[NETWORK_LAYER_SIZE[i]];
            this.bias[i] = new double[NETWORK_LAYER_SIZE[i]];

            if(i > 0){
                weights[i] = new double[NETWORK_LAYER_SIZE[i]][NETWORK_LAYER_SIZE[i-1]];
            }
        }

    }

    public static void main(String[] args){
        Network net = new Network(4,3,2,3,4);
    }

}
