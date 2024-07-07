#include <iostream>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <map>
#include <string>
#include <dirent.h>  // For POSIX directory traversal
#include <cstring>   // For strcmp
#include <algorithm>

#include "yolov8.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"
#include "postprocess.h"
#include "cosine_similarity.h"

#if defined(RV1106_1103) 
    #include "dma_alloc.hpp"
#endif

static rknn_app_context_t rknn_app_ctx;
static face_rec_result fr_result;

struct Image {
    std::string path;
    std::string label;
};

std::vector<Image> load_images(const std::string& directory) {
    std::vector<Image> images;
    DIR* dirp = opendir(directory.c_str());
    if (dirp == nullptr) {
        perror("opendir");
        return images;
    }

    struct dirent* dp;
    while ((dp = readdir(dirp)) != nullptr) {
        std::string filename = dp->d_name;
        if (filename == "." || filename == "..") {
            continue;
        }
        std::string path = directory + "/" + filename;
        std::string label = filename.substr(0, filename.find('.'));
        images.push_back({ path, label });
    }

    closedir(dirp);
    return images;
}

static void get_embedding(const std::string & image_path, std::array<float, 512> & embedding)
{
    image_buffer_t src_image;
    memset(&src_image, 0, sizeof(image_buffer_t));
    read_image(image_path.c_str(), &src_image);

    //RV1106 rga requires that input and output bufs are memory allocated by dma
    dma_buf_alloc(RV1106_CMA_HEAP_PATH, src_image.size, &rknn_app_ctx.img_dma_buf.dma_buf_fd, 
                    (void **) & (rknn_app_ctx.img_dma_buf.dma_buf_virt_addr));
    memcpy(rknn_app_ctx.img_dma_buf.dma_buf_virt_addr, src_image.virt_addr, src_image.size);
    dma_sync_cpu_to_device(rknn_app_ctx.img_dma_buf.dma_buf_fd);
    free(src_image.virt_addr);
    src_image.virt_addr = (unsigned char *)rknn_app_ctx.img_dma_buf.dma_buf_virt_addr;
    src_image.fd = rknn_app_ctx.img_dma_buf.dma_buf_fd;
    rknn_app_ctx.img_dma_buf.size = src_image.size;


    inference_arcface_model(&rknn_app_ctx, &src_image, &fr_result);
    if (src_image.virt_addr != NULL)
    {
        dma_buf_free(rknn_app_ctx.img_dma_buf.size, &rknn_app_ctx.img_dma_buf.dma_buf_fd, 
                rknn_app_ctx.img_dma_buf.dma_buf_virt_addr);
    }
    embedding = fr_result.embedding;
}

static float compute_rank1_accuracy(const std::vector<Image>& gallery, const std::vector<Image>& probes) {
    int correct_matches = 0;

    // Precompute embeddings for the gallery set
    std::map<std::string, std::array<float, 512>> gallery_embeddings;
    for (const auto& img : gallery) {
        get_embedding(img.path, gallery_embeddings[img.path]);
    }

    // Process each probe image
    for (const auto& probe : probes) {
        std::array<float, 512> probe_embedding;
        get_embedding(probe.path, probe_embedding);

        // Find the best match in the gallery
        std::string best_match;
        float best_similarity = -1.0;
        for (const auto& gallery_img : gallery) {
            float similarity = cosineSimilarity(probe_embedding, gallery_embeddings[gallery_img.path]);
            if (similarity > best_similarity) {
                best_similarity = similarity;
                best_match = gallery_img.label;
            }
        }

        // Check if the best match is correct
        if (best_match == probe.label) {
            correct_matches++;
        }
    }

    return static_cast<float>(correct_matches) / probes.size();
}

template <int N>
static float compute_rankN_accuracy(const std::vector<Image>& gallery, const std::vector<Image>& probes) {
    int correct_matches = 0;

    // Precompute embeddings for the gallery set
    std::map<std::string, std::array<float, 512>> gallery_embeddings;
    for (const auto& img : gallery) {
        get_embedding(img.path, gallery_embeddings[img.path]);
    }

    // Process each probe image
    for (const auto& probe : probes) {
        std::array<float, 512> probe_embedding;
        get_embedding(probe.path, probe_embedding);

        // Find the best match in the gallery
        std::vector<std::pair<float, std::string>> similarities;
        for (const auto& gallery_img : gallery) {
            float similarity = cosineSimilarity(probe_embedding, gallery_embeddings[gallery_img.path]);
            similarities.push_back({ similarity, gallery_img.label });
        }

        std::sort(similarities.begin(), similarities.end(), std::greater<>());

        // Check if the best N matches contains correct label
        bool found = false;
        for (int i = 0; i < std::min(N, static_cast<int>(similarities.size())); i++) {
            if (similarities[i].second == probe.label) {
                found = true;
                break;
            }
        }

        if (found) {
            correct_matches++;
        }
    }

    return static_cast<float>(correct_matches) / probes.size();
}

int main(int argc, char **argv)
{
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
    init_arcface_model("model/w600k_mbf.rknn", &rknn_app_ctx);
    
    std::string dataset_directory = "eval_dataset/yale_face";
    std::vector<Image> all_images = load_images(dataset_directory);

    // for (auto & image : all_images) {
    //     std::cout << image.path << std::endl;
    //     std::cout << image.label << std::endl;
    // }

    std::vector<Image> gallery;
    std::vector<Image> probes;

    for (const auto& img : all_images) {
        if (img.path.find(".normal") != std::string::npos) {
            gallery.push_back(img);
        } else {
            probes.push_back(img);
        }
    }

    float acc_r1 = compute_rank1_accuracy(gallery, probes);
    float acc_r2 = compute_rankN_accuracy<2>(gallery, probes);
    float acc_r3 = compute_rankN_accuracy<3>(gallery, probes);

    printf("Rank 1 Accuracy is: %f\n", acc_r1);
    printf("Rank 2 Accuracy is: %f\n", acc_r2);
    printf("Rank 3 Accuracy is: %f\n", acc_r3);

    release_arcface_model(&rknn_app_ctx);

    return 0;
}