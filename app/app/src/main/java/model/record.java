package model;

public class record {
    private String id;
    private String date;
    private float prediction;
    private String original_image;
    private String type;
    private String segment_image;

    public record(String id, String date, float prediction, String original_image, String type, String segment_image) {
        this.id = id;
        this.date = date;
        this.prediction = prediction;
        this.original_image = original_image;
        this.type = type;
        this.segment_image = segment_image;
    }

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getDate() {
        return date;
    }

    public void setDate(String date) {
        this.date = date;
    }

    public float getPrediction() {
        return prediction;
    }

    public void setPrediction(float prediction) {
        this.prediction = prediction;
    }

    public String getOriginal_image() {
        return original_image;
    }

    public void setOriginal_image(String original_image) {
        this.original_image = original_image;
    }

    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type;
    }

    public String getSegment_image() {
        return segment_image;
    }

    public void setSegment_image(String segment_image) {
        this.segment_image = segment_image;
    }
}
