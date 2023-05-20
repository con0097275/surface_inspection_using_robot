package com.example.demoiot;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import model.ImageName;
import model.record;
import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class Connectivity {
    public static String geturl (String url_esp32){

        OkHttpClient client = new OkHttpClient();
        Request request = new Request.Builder()
                .url(url_esp32)
                .build();
        try  {
            Response response = client.newCall(request).execute();
            return response.body().string();

        } catch (IOException error) {
            return null;
        }


    }
    public static String postimage (String url,String base64string) {

//       OkHttpClient client = new OkHttpClient();
//
//        RequestBody formBody = new MultipartBody.Builder()
//                .setType(MultipartBody.FORM)
//                .addFormDataPart("image", base64string)
//                .build();
//
//        Request request = new Request.Builder()
//                .url(url)
//                .post(formBody)
//                .build();
        OkHttpClient client = new OkHttpClient().newBuilder().build();
        MediaType mediaType = MediaType.parse("application/json");
        RequestBody body = RequestBody.create(mediaType,
                "{\n" +
                        "    \"image\":\""+base64string+"\"\n" +
                        "}");

        Request request = new Request.Builder()
                .url(url)
                .method("POST", body)
                .addHeader("Content-Type", "application/json")
                .build();
        Response response = null;
        try {
            response = client.newCall(request).execute();
            if (!response.isSuccessful()) throw new IOException("Unexpected code " + response);
            String responsebody = response.body().string();
            JSONObject jObject = new JSONObject(responsebody);
            return jObject.getString("image");
        } catch (IOException e) {
            System.out.println("loi io");
            throw new RuntimeException(e);
        } catch (JSONException e) {
            System.out.println("loi json");
            throw new RuntimeException(e);
        } finally {
            if(response!=null){
                response.close();
            }
        }
    }

    public static List<record> getIdAndType (){
        OkHttpClient client = new OkHttpClient().newBuilder().build();
        MediaType mediaType = MediaType.parse("application/json");
        RequestBody body = RequestBody.create(mediaType,
                "{\n" +
                        "    \"collection\":\"fault_detection\",\n" +
                        "    \"database\":\"thesis\",\n" +
                        "    \"dataSource\":\"Cluster0\",\n" +
                        "    \"projection\": {\n" +
                        "      \"_id\": 1,\n" +
                        "      \"type\": 1,\n" +
                        "      \"segment_image\": 1\n" +
                        "    }\n" +
                        "}");

        Request request = new Request.Builder()
                .url("https://ap-southeast-1.aws.data.mongodb-api.com/app/data-wlatu/endpoint/data/v1/action/find")
                .method("POST", body)
                .addHeader("Content-Type", "application/json")
                .addHeader("Access-Control-Request-Headers", "*")
                .addHeader("api-key", "LFyT8MWcEraGxtCsMJpceBO8q72WLX8mInon25j6kbVCgv2j5vSwVYzNVzdxFsqh")
                .build();
        Response response=null;
        try {
            response = client.newCall(request).execute();
            if (!response.isSuccessful()) throw new IOException("Unexpected code " + response);
            String responsebody =response.body().string();
            JSONObject myjson = new JSONObject(responsebody);
            JSONArray the_json_array = myjson.getJSONArray("documents");
            int size = the_json_array.length();
            ArrayList<record> nameList = new ArrayList<record>();
            for (int i = 0; i < size; i++) {
                JSONObject another_json_object = the_json_array.getJSONObject(i);
                nameList.add(new record(another_json_object.getString("_id"),"",
                        0f,"",another_json_object.getString("type"),
                        another_json_object.getString("segment_image")));
            }
            return nameList;
        } catch (IOException e) {
            System.out.println("loi io");
            throw new RuntimeException(e);
        } catch (Exception e) {
            System.out.println("loi json");
            throw new RuntimeException(e);
        } finally {
            if(response!=null) {
                response.close();
            }
        }
//        Response response = client.newCall(request).execute();
//        if (!response.isSuccessful()) throw new IOException("Unexpected code " + response);
//
//        String responsebody =response.body().string();
//        JSONObject myjson = new JSONObject(responsebody);
//        JSONArray the_json_array = myjson.getJSONArray("documents");
//        int size = the_json_array.length();
//        ArrayList<record> nameList = new ArrayList<record>();
//        for (int i = 0; i < size; i++) {
//            JSONObject another_json_object = the_json_array.getJSONObject(i);
//            nameList.add(new record(another_json_object.getString("_id"),"",
//                    0f,"",another_json_object.getString("type"),""));
//        }
//        return nameList;
    }

    public static List<ImageName> getImageNames () throws IOException, JSONException {
        OkHttpClient client = new OkHttpClient().newBuilder().build();
        MediaType mediaType = MediaType.parse("application/json");
        RequestBody body = RequestBody.create(mediaType,
                "{\n" +
                        "    \"collection\":\"fault_detection\",\n" +
                        "    \"database\":\"thesis\",\n" +
                        "    \"dataSource\":\"Cluster0\",    \n" +
                        "    \"projection\": {\n" +
                        "      \"_id\": 1\n" +
                        "    }\n" +
                        "  \n" +
                        "}");

        Request request = new Request.Builder()
                .url("https://ap-southeast-1.aws.data.mongodb-api.com/app/data-wlatu/endpoint/data/v1/action/find")
                .method("POST", body)
                .addHeader("Content-Type", "application/json")
                .addHeader("Access-Control-Request-Headers", "*")
                .addHeader("api-key", "LFyT8MWcEraGxtCsMJpceBO8q72WLX8mInon25j6kbVCgv2j5vSwVYzNVzdxFsqh")
                .build();
        Response response = client.newCall(request).execute();
        if (!response.isSuccessful()) throw new IOException("Unexpected code " + response);

        String responsebody =response.body().string();
        //JSONObject jObject = new JSONObject(responsebody);
        JSONObject myjson = new JSONObject(responsebody);
        JSONArray the_json_array = myjson.getJSONArray("documents");
        int size = the_json_array.length();
        //ArrayList<JSONObject> arrays = new ArrayList<JSONObject>();
        ArrayList<ImageName> nameList = new ArrayList<ImageName>();
        for (int i = 0; i < size; i++) {
            JSONObject another_json_object = the_json_array.getJSONObject(i);
            //arrays.add(another_json_object);
            nameList.add(new ImageName(another_json_object.getString("_id")));
        }
        return nameList;
    }

}
