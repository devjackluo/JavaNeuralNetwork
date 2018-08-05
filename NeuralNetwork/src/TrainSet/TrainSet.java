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
            for(int k = 0; k < INPUT_SIZE; k++){
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

        StringBuilder sb = new StringBuilder();
        int index = 0;
        for(double[][] d : data){
            sb.append(index + ":  "+ Arrays.toString(d[0]) +"  >-||-<  "+Arrays.toString(d[1]) +"\n");
            index++;
        }

        return sb.toString();

    }

}
