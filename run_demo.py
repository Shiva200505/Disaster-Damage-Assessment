from member5_evaluation.inference_pipeline import run_inference

if __name__ == "__main__":
    print("Running end-to-end inference for Hurricane Harvey 0192...")
    result = run_inference(
        pre_img_path    = "data/xbd/test/images/hurricane-harvey_00000192_pre_disaster.png",
        post_img_path   = "data/xbd/test/images/hurricane-harvey_00000192_post_disaster.png",
        geojson_path    = "data/xbd/test/labels/hurricane-harvey_00000192_post_disaster.json",
        siamese_ckpt    = "checkpoints/siamese_best.pth",
        output_dir      = "outputs/demo",
        disaster_name   = "Hurricane Harvey",
    )
    print("Inference completed successfully!")
    print(f"Metrics: {result.get('metrics', {})}")
