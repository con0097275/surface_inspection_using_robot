package com.example.demoiot;

import android.content.Intent;
import android.os.AsyncTask;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;

import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import java.util.ArrayList;
import java.util.List;

import model.record;

public class ImageListActivity extends AppCompatActivity {
    private ArrayList<record> mHeros ;
    private RecyclerView mRecyclerHero;
    private ImageNameAdapter mHeroAdapter ;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_hero);
        mRecyclerHero = findViewById(R.id.recyclerHero);
        Button btnmain= findViewById(R.id.button10);
        btnmain.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(ImageListActivity.this, MainActivity.class);
                startActivity(intent);
            }

        });
        Button btnchart= findViewById(R.id.button12);
        btnchart.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(ImageListActivity.this, chart_activity.class);
                startActivity(intent);
            }

        });

        Button btncapture= findViewById(R.id.button13);
        btncapture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(ImageListActivity.this, ImageCaptureActivity.class);
                startActivity(intent);
            }

        });
        new DownloadImageNameTask().execute();
    }
    private class DownloadImageNameTask extends AsyncTask<Void, Void, List<record>> {

        protected List<record> doInBackground(Void... voids) {

            List<record> resultImage = null;
            resultImage=Connectivity.getIdAndType();

            return resultImage;
        }
        protected void onPostExecute(List<record> result) {
            mHeros= (ArrayList<record>) result;
            mHeroAdapter = new ImageNameAdapter(ImageListActivity.this,mHeros);
            mRecyclerHero.setAdapter(mHeroAdapter);
            mRecyclerHero.setLayoutManager(new LinearLayoutManager(ImageListActivity.this));
        }
    }
}
