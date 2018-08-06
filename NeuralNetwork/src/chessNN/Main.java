package chessNN;

import FullyConnectedNetwork.Network;
import FullyConnectedNetwork.NetworkTools;
import TrainSet.TrainSet;
import mnist.MnistImageFile;
import mnist.MnistLabelFile;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Scanner;
import java.util.stream.Stream;

public class Main {

    public static int count;

    public static void main(String[] args){


        Network network = null;

        if(Files.exists(Paths.get("./res/chess/models/test.model"))){
            try {
                network = Network.loadNetwork("./res/chess/models/test.model");
                System.out.println("LOADED");
            } catch (Exception e) {
                e.printStackTrace();
            }
        }else{
            network = new Network(70, 60, 50, 40, 30, 20, 20, 30, 40, 50, 60, 70);
            System.out.println("CREATED NEW NETWORK");
        }

        TrainSet set = createTrainSet();
        //System.out.println(count);
        trainData(network, set, 200, 50, 10000);


        TrainSet testSet = createTrainSet();
        testTrainSet(network, testSet, 100);


    }


    public static TrainSet createTrainSet() {

        TrainSet set = new TrainSet(70, 70);

        try {

            File file = new File("./res/chess/mini.txt");
            Scanner fileInput = new Scanner(file);

            while(fileInput.hasNext()) {

                double[] input = new double[70];
                double[] output = new double[70];

                //the label has a value 0-9

                String currentData = fileInput.nextLine();
                String[] data = currentData.split("=");

                //the position of move order in which the best move resides at
                output[Integer.parseInt(data[1])] = 1d;



                String[] items = data[0].replaceAll("\\[", "")
                        .replaceAll("\\]", "")
                        .replaceAll("\\s", "")
                        .split(",");

                //System.out.println(data[0]);
                //count++;

                int[] results = new int[items.length];
                for (int i = 0; i < items.length; i++) {
                    try {
                        results[i] = Integer.parseInt(items[i]);
                    } catch (NumberFormatException nfe) {
                        //NOTE: write something here if you need to recover from formatting errors
                    }
                }


//                if(results.length < 20) {
//                    count++;
//                }

//                if(Integer.parseInt(data[1]) > count){
//                    count = Integer.parseInt(data[1]);
//                }

                //count += Integer.parseInt(data[1]);


                for(int j = 0; j < results.length; j++){
                    //The read() method increments the file pointer to point to the next byte in the file after the byte just read!
                    input[j] = (double)results[j]/ (double)450;
                }

                //System.out.println(Arrays.toString(input));

                set.addData(input, output);

            }


        } catch (Exception e) {
            e.printStackTrace();
        }

        System.out.println("DONE PREPARING SET");

        return set;
    }


    public static TrainSet createTestSet() {

        TrainSet set = new TrainSet(70, 70);

        try {

            File file = new File("./res/chess/mini.txt");
            Scanner fileInput = new Scanner(file);

            for(int d = 0; d < 5000; d++) {

                for(int r = 0; r < ((int)(Math.random()*10)); r++){
                    fileInput.nextLine();
                }

                double[] input = new double[70];
                double[] output = new double[70];

                //the label has a value 0-9

                String currentData = fileInput.nextLine();
                String[] data = currentData.split("=");

                output[Integer.parseInt(data[1])] = 1d;

                String[] items = data[0].replaceAll("\\[", "")
                        .replaceAll("\\]", "")
                        .replaceAll("\\s", "")
                        .split(",");

                int[] results = new int[items.length];
                for (int i = 0; i < items.length; i++) {
                    try {
                        results[i] = Integer.parseInt(items[i]);
                    } catch (NumberFormatException nfe) {
                        //NOTE: write something here if you need to recover from formatting errors
                    }
                }

                for(int j = 0; j < results.length; j++){
                    //The read() method increments the file pointer to point to the next byte in the file after the byte just read!
                    input[j] = (double)results[j]/ (double)450;
                }

                //System.out.println(Arrays.toString(input));

                set.addData(input, output);

            }


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

            try {
                net.saveNetwork("./res/chess/models/test.model");
                //System.out.println("SAVED NETWORK");
            } catch (Exception ex) {
                ex.printStackTrace();
            }

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
                System.out.println(Arrays.toString( net.calculate(set.getInput(i))) + ": \n         " + Arrays.toString(set.getOutput(i)));
            }
            */

        }
        System.out.println("Testing finished, RESULT: " + correct + " / " + set.size()+ "  -> " + (double)correct / (double)set.size() +" %");
    }



}
