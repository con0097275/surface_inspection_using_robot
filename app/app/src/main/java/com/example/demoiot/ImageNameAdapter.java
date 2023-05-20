package com.example.demoiot;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Base64;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import com.bumptech.glide.Glide;

import java.util.ArrayList;

import model.record;

public class ImageNameAdapter extends
        RecyclerView.Adapter<ImageNameAdapter.ViewHolder> {

    public class ViewHolder extends RecyclerView.ViewHolder {
        private ImageView mImageHero;
        private TextView mTextName;

        public ViewHolder(@NonNull View itemView) {
            super(itemView);
            mImageHero = itemView.findViewById(R.id.image_hero);
            mTextName = itemView.findViewById(R.id.text_name);
        }
    }
    private Context mContext;
    private ArrayList<record> mImagenames;

    public ImageNameAdapter(Context mContext, ArrayList<record> mImagenames) {
        this.mContext = mContext;
        this.mImagenames = mImagenames;
    }
    @NonNull
    @Override
    public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        LayoutInflater inflater = LayoutInflater.from(mContext);
        View heroView = inflater.inflate(R.layout.item_hero, parent, false);
        ViewHolder viewHolder = new ViewHolder(heroView);
        return viewHolder;
    }

    @Override
    public void onBindViewHolder(@NonNull ViewHolder holder, int position) {
        record hero = mImagenames.get(position);
        byte[] bytes = Base64.decode(hero.getSegment_image(), Base64.DEFAULT);
        Bitmap bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.length);
        Glide.with(mContext)
                .load(bitmap)
                .into(holder.mImageHero);
        holder.mTextName.setText(hero.getId());
    }

    @Override
    public int getItemCount() {
        return mImagenames.size();
    }

}


