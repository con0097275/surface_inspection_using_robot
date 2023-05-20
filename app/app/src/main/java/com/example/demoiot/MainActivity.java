package com.example.demoiot;

import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import com.github.angads25.toggle.interfaces.OnToggledListener;
import com.github.angads25.toggle.model.ToggleableView;
import com.github.angads25.toggle.widget.LabeledSwitch;

import org.eclipse.paho.client.mqttv3.IMqttDeliveryToken;
import org.eclipse.paho.client.mqttv3.MqttCallbackExtended;
import org.eclipse.paho.client.mqttv3.MqttException;
import org.eclipse.paho.client.mqttv3.MqttMessage;

import java.nio.charset.Charset;

public class MainActivity extends AppCompatActivity {
    MQTTHelper mqttHelper;

    TextView txtTemp, txtHumi;
    LabeledSwitch btnLED;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);


        setContentView(R.layout.activity_main);
        txtTemp= findViewById(R.id.txtTemperature);
        txtHumi= findViewById(R.id.txtHumidity);
        btnLED= findViewById(R.id.btnLED);


        btnLED.setOnToggledListener(new OnToggledListener() {
            @Override
            public void onSwitched(ToggleableView toggleableView, boolean isOn) {
                if (isOn == true) {
                    sendDataMQTT("van00972751/feeds/nutnhan1","1");
                } else{
                    sendDataMQTT("van00972751/feeds/nutnhan1","0");
                }

            }
        });



        Button btnhistory= findViewById(R.id.button11);
        btnhistory.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(MainActivity.this, ImageListActivity.class);
                startActivity(intent);
            }

        });
        Button btnchart= findViewById(R.id.button12);
        btnchart.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(MainActivity.this, chart_activity.class);
                startActivity(intent);
            }

        });

        Button btncapture= findViewById(R.id.button13);
        btncapture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(MainActivity.this, ImageCaptureActivity.class);
                startActivity(intent);
            }

        });



        startMQTT();

//        SwitchLayoutBinding binding = SwitchLayoutBinding.inflate(getLayoutInflater());
//        binding.materialSwitch.setOnCheckedChangeListener((buttonView, isChecked) -> {
//            if (isChecked) {
//                // The switch is checked.
//            } else {
//                // The switch isn't checked.
//            }
//        });
    }


    public void sendDataMQTT(String topic, String value){
        MqttMessage msg = new MqttMessage();
        msg.setId(1234);
        msg.setQos(0);
        msg.setRetained(false);

        byte[] b = value.getBytes(Charset.forName("UTF-8"));
        msg.setPayload(b);

        try {
            mqttHelper.mqttAndroidClient.publish(topic, msg);
        }catch (MqttException e){
        }
    }

    public void startMQTT() {
        mqttHelper = new MQTTHelper(this);
        mqttHelper.setCallback(new MqttCallbackExtended() {
            @Override
            public void connectComplete(boolean reconnect, String serverURI) {

            }

            @Override
            public void connectionLost(Throwable cause) {

            }

            @Override
            public void messageArrived(String topic, MqttMessage message) throws Exception {
                Log.d("TEST", topic + "***" + message.toString());
                if (topic.contains("cambien1")) {
                    txtTemp.setText(message.toString() + "°C");
                } else if (topic.contains("cambien2")) {
                    txtHumi.setText(message.toString() + "%");
                } else if (topic.contains("nutnhan1")) {
                    if (message.toString().equals("1")) {
                        btnLED.setOn(true);
                    } else{
                        btnLED.setOn(false);
                    }
                }
            }

            @Override
            public void deliveryComplete(IMqttDeliveryToken token) {

            }
        });
    }
}

