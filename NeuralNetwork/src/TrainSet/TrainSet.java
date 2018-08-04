package TrainSet;

import java.util.ArrayList;
import java.util.Arrays;

import static FullyConnectedNetwork.NetworkTools.randomIntArray;

public class TrainSet {


    public final int INPUT_SIZE;
    public final int OUTPUT_SIZE;

    private ArrayList<double[][]> data = new ArrayList<>();

    public TrainSet(int INPUT_SIZE, int OUTPUT_SIZE){
        this.INPUT_SIZE = INPUT_SIZE;
        this.OUTPUT_SIZE = OUTPUT_SIZE;
    }

    public void addSampleData(int amount){
        for(int i = 0; i < amount; i++){
            double[] a = new double[INPUT_SIZE];
            double[] b = new double[OUTPUT_SIZE];
            for(int k = 0; k < 4; k++){
                //size of input
                a[k] = (double) ((int)(Math.random()*10)) / (double)10;
                //size of output, kind of only works if output is smaller hmm
                if(k < OUTPUT_SIZE){
                    b[k] = (double) ((int)(Math.random()*10)) / (double)10;
                }

            }
            addData(a,b);
        }
    }

    public void addData(double[] in, double[] expected){
        if(in.length != INPUT_SIZE || expected.length != OUTPUT_SIZE){
            return;
        }
        //an array of double[]s
        data.add(new double[][]{in, expected});
    }

    public TrainSet extractBatch(int size){
        if(size > 0 && size <= this.size()) {
            TrainSet set = new TrainSet(INPUT_SIZE, OUTPUT_SIZE);
            Integer[] ids = randomIntArray(0,this.size() - 1, size);
            for(Integer i:ids) {
                set.addData(this.getInput(i),this.getOutput(i));
            }
            return set;

        }else {
            return this;
        }
    }

    public double[] getInput(int index){
        if(index >= 0 && index < size()){
            return data.get(index)[0];
        }
        return null;
    }

    public double[] getOutput(int index){
        if(index >= 0 && index < size()){
            return data.get(index)[1];
        }
        return null;
    }

    public int size(){
        return data.size();
    }

    @Override
    public String toString() {
        /*
        String s = "TrainSet ["+INPUT_SIZE+ " ; "+OUTPUT_SIZE+"]\n";
        int index = 0;

        for(double[][] r:data) {
            s += index +":   "+ Arrays.toString(r[0]) +"  >-||-<  "+Arrays.toString(r[1]) +"\n";
            index++;
        }
        return s;
        */


        StringBuilder sb = new StringBuilder();
        int index = 0;
        for(double[][] d : data){
            sb.append(index + ":  "+ Arrays.toString(d[0]) +"  >-||-<  "+Arrays.toString(d[1]) +"\n");
            index++;
        }

        return sb.toString();

    }

    public static void main(String[] args){
        TrainSet set = new TrainSet(4, 4);

        for(int i = 0; i < 8; i++){
            double[] a = new double[4];
            double[] b = new double[4];
            for(int k = 0; k < 4; k++){
                //size of input
                a[k] = (double) ((int)(Math.random()*10)) / (double)10;
                //size of output
                if(k < 4){
                    b[k] = (double) ((int)(Math.random()*10)) / (double)10;
                }

            }
            set.addData(a,b);
        }

        System.out.println(set);
        System.out.println(set.extractBatch(3));

    }


//
//    public final int INPUT_SIZE;
//    public final int OUTPUT_SIZE;
//
//    //double[][] <- index1: 0 = input, 1 = output || index2: index of element
//    private ArrayList<double[][]> data = new ArrayList<>();
//
//    public TrainSet(int INPUT_SIZE, int OUTPUT_SIZE) {
//        this.INPUT_SIZE = INPUT_SIZE;
//        this.OUTPUT_SIZE = OUTPUT_SIZE;
//    }
//
//    public void addData(double[] in, double[] expected) {
//        if(in.length != INPUT_SIZE || expected.length != OUTPUT_SIZE) return;
//        data.add(new double[][]{in, expected});
//    }
//
//    public TrainSet extractBatch(int size) {
//        if(size > 0 && size <= this.size()) {
//            TrainSet set = new TrainSet(INPUT_SIZE, OUTPUT_SIZE);
//            Integer[] ids = randomIntArray(0,this.size() - 1, size);
//            for(Integer i:ids) {
//                set.addData(this.getInput(i),this.getOutput(i));
//            }
//            return set;
//        }else return this;
//    }
//
//    public static void main(String[] args) {
//        TrainSet set = new TrainSet(3,2);
//
//        for(int i = 0; i < 8; i++) {
//            double[] a = new double[3];
//            double[] b = new double[2];
//            for(int k = 0; k < 3; k++) {
//                a[k] = (double)((int)(Math.random() * 10)) / (double)10;
//                if(k < 2) {
//                    b[k] = (double)((int)(Math.random() * 10)) / (double)10;
//                }
//            }
//            set.addData(a,b);
//        }
//
//        System.out.println(set);
//        System.out.println(set.extractBatch(3));
//    }
//
//    public String toString() {
//        String s = "TrainSet ["+INPUT_SIZE+ " ; "+OUTPUT_SIZE+"]\n";
//        int index = 0;
//        for(double[][] r:data) {
//            s += index +":   "+Arrays.toString(r[0]) +"  >-||-<  "+Arrays.toString(r[1]) +"\n";
//            index++;
//        }
//        return s;
//    }
//
//    public int size() {
//        return data.size();
//    }
//
//    public double[] getInput(int index) {
//        if(index >= 0 && index < size())
//            return data.get(index)[0];
//        else return null;
//    }
//
//    public double[] getOutput(int index) {
//        if(index >= 0 && index < size())
//            return data.get(index)[1];
//        else return null;
//    }
//
//    public int getINPUT_SIZE() {
//        return INPUT_SIZE;
//    }
//
//    public int getOUTPUT_SIZE() {
//        return OUTPUT_SIZE;
//    }


}
