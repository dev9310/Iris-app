package com.example.iris;

import androidx.appcompat.app.AppCompatActivity;
import java.nio.ByteBuffer;
import android.os.Bundle;
import android.view.View;

import com.example.iris.databinding.ActivityMainBinding;
import com.example.iris.ml.Iris;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;

public class MainActivity extends AppCompatActivity {

    ActivityMainBinding binding;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityMainBinding.inflate(getLayoutInflater());

        setContentView(binding.getRoot());


        binding.btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                Float v1 = Float.valueOf(binding.t1.getText().toString());
                Float v2 = Float.valueOf(binding.t2.getText().toString());
                Float v3 = Float.valueOf(binding.t3.getText().toString());
                Float v4 = Float.valueOf(binding.t4.getText().toString());

                ByteBuffer byteBuffer = ByteBuffer.allocate(4*4);
                byteBuffer.putFloat(v1);
                byteBuffer.putFloat(v2);
                byteBuffer.putFloat(v3);
                byteBuffer.putFloat(v4);


                try {
                    Iris model = Iris.newInstance(MainActivity.this);

                    // Creates inputs for reference.
                    TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 4}, DataType.FLOAT32);
                    inputFeature0.loadBuffer(byteBuffer);

                    // Runs model inference and gets result.
                    Iris.Outputs outputs = model.process(inputFeature0);
//            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
                    TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
                    float[] outputArray = outputFeature0.getFloatArray();


                    binding.result.setText("Iris Satosa" + outputArray[0] + "\n" + "versicolor" + outputArray[1] + "\n" + "verginica" +outputArray[2] + "\n"  );
                    // Releases model resources if no longer used.
                    model.close();
                } catch (IOException e) {
                    // TODO Handle the exception
                }

            }
        });

    }
}