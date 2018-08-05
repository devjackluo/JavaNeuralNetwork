package mnist;

import FullyConnectedNetwork.Network;
import FullyConnectedNetwork.NetworkTools;
import TrainSet.TrainSet;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Paths;

/**
 * Created by Luecx on 10.08.2017.
 */
public class Mnist {

    public static void main(String[] args) {

        Network network = null;

        if(Files.exists(Paths.get("./res/models/test.model"))){
            try {
                network = Network.loadNetwork("./res/models/test.model");
                System.out.println("LOADED");
            } catch (Exception e) {
                e.printStackTrace();
            }
        }else{
            network = new Network(784, 392, 196, 10);
            System.out.println("CREATED NEW NETWORK");
        }

        TrainSet set = createTrainSet(0,29999, "/res/train-images.idx3-ubyte", "/res/train-labels.idx1-ubyte");
        trainData(network, set, 10, 50, 1000);

        TrainSet testSet = createTrainSet(0,4999, "/res/t10k-images.idx3-ubyte", "/res/t10k-labels.idx1-ubyte");
        testTrainSet(network, testSet, 100);

        try {
            network.saveNetwork("./res/models/test.model");
            System.out.println("SAVED NETWORK");
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    public static TrainSet createTrainSet(int start, int end, String inputs, String outputs) {

        TrainSet set = new TrainSet(28 * 28, 10);

        try {

            String path = new File("").getAbsolutePath();

            MnistImageFile m = new MnistImageFile(path + inputs, "r");
            MnistLabelFile l = new MnistLabelFile(path + outputs, "r");

            for(int i = start; i <= end; i++) {
                /*
                if(i % 100 ==  0){
                    System.out.println("prepared: " + i);
                }
                */

                double[] input = new double[28 * 28];
                double[] output = new double[10];

                //the label has a value 0-9
                output[l.readLabel()] = 1d;
                for(int j = 0; j < 28*28; j++){
                    //The read() method increments the file pointer to point to the next byte in the file after the byte just read!
                    input[j] = (double)m.read() / (double)256;
                }

                set.addData(input, output);
                m.next();
                l.next();
            }

            m.close();
            l.close();

        } catch (Exception e) {
            e.printStackTrace();
        }

        System.out.println("DONE PREPARING SET");

        return set;
    }

    public static void trainData(Network net,TrainSet set, int epochs, int loops, int batch_size) {
        for(int e = 0; e < epochs;e++) {
            net.trainSet(set, loops, batch_size);
            System.out.println(">>>>>>>>>>>>>>>>>>>>>>>>>   "+ e+ "   <<<<<<<<<<<<<<<<<<<<<<<<<<");
        }
    }

    public static void testTrainSet(Network net, TrainSet set, int printSteps) {
        int correct = 0;
        for(int i = 0; i < set.size(); i++) {

            double highest = NetworkTools.indexOfHighestValue(net.calculate(set.getInput(i)));
            double actualHighest = NetworkTools.indexOfHighestValue(set.getOutput(i));
            if(highest == actualHighest) {

                correct ++ ;
            }
            /*
            if(i % printSteps == 0) {
                System.out.println(i + ": " + (double)correct / (double) (i + 1));
            }
            */
        }
        System.out.println("Testing finished, RESULT: " + correct + " / " + set.size()+ "  -> " + (double)correct / (double)set.size() +" %");
    }
}
