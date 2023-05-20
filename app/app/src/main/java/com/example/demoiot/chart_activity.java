package com.example.demoiot;

import android.content.Intent;
import android.os.AsyncTask;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;

import androidx.appcompat.app.AppCompatActivity;

import com.anychart.AnyChart;
import com.anychart.AnyChartView;
import com.anychart.chart.common.dataentry.DataEntry;
import com.anychart.chart.common.dataentry.ValueDataEntry;
import com.anychart.charts.Cartesian;
import com.anychart.core.cartesian.series.Column;
import com.anychart.enums.Anchor;
import com.anychart.enums.HoverMode;
import com.anychart.enums.Position;
import com.anychart.enums.TooltipPositionMode;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import model.record;

public class chart_activity extends AppCompatActivity {
    Cartesian cartesian;
    AnyChartView anyChartView;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.chart_activity);

        anyChartView = findViewById(R.id.any_chart_view);
        anyChartView.setProgressBar(findViewById(R.id.progress_bar));
        cartesian = AnyChart.column();

        Button btnmain= findViewById(R.id.button6);
        btnmain.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(chart_activity.this, MainActivity.class);
                startActivity(intent);
            }

        });
        Button btncapture= findViewById(R.id.button9);
        btncapture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(chart_activity.this, ImageCaptureActivity.class);
                startActivity(intent);
            }

        });

        Button btnhistory= findViewById(R.id.button7);
        btnhistory.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(chart_activity.this, ImageListActivity.class);
                startActivity(intent);
            }

        });

        new DownloadChartData().execute();

    }

    private class DownloadChartData extends AsyncTask<Void, Void, List<DataEntry>> {

        protected List<DataEntry> doInBackground(Void... voids) {
            List<DataEntry> data = new ArrayList<>();
            List<record> records = null;
            records=Connectivity.getIdAndType();
            Map<String,Integer> chartData = new HashMap<>();
            for (record onerecord:
                 records) {
                if(chartData.containsKey(onerecord.getType())){
                    chartData.put(onerecord.getType(),chartData.get(onerecord.getType())+1);
                }else{
                    chartData.put(onerecord.getType(),1);
                }
            }
            Set<String> set = chartData.keySet();
            for (String key : set) {
                data.add(new ValueDataEntry(key,chartData.get(key)));
            }
            return data;
        }

        protected void onPostExecute(List<DataEntry> data) {
            Column column = cartesian.column(data);

            column.tooltip()
                    .titleFormat("{%X}")
                    .position(Position.CENTER_BOTTOM)
                    .anchor(Anchor.CENTER_BOTTOM)
                    .offsetX(0d)
                    .offsetY(5d)
                    .format("{%Value}{groupsSeparator: }");

            cartesian.animation(true);
            cartesian.title("Lỗi bề mặt");

            cartesian.yScale().minimum(0d);

            cartesian.yAxis(0).labels().format("{%Value}{groupsSeparator: }");

            cartesian.tooltip().positionMode(TooltipPositionMode.POINT);
            cartesian.interactivity().hoverMode(HoverMode.BY_X);

            cartesian.xAxis(0).title("Loại lỗi");
            cartesian.yAxis(0).title("Tần suất");

            anyChartView.setChart(cartesian);
        }
    }
}
