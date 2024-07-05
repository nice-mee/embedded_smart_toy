// Copyright (c) 2023 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "yolov8.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"
#include "postprocess.h"
#include "cosine_similarity.h"

#if defined(RV1106_1103) 
    #include "dma_alloc.hpp"
#endif

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char **argv)
{
    if (argc != 4)
    {
        printf("%s <model_path> <image_path>\n", argv[0]);
        return -1;
    }

    const char *model_path = argv[1];
    const char *image1_path = argv[2];
    const char *image2_path = argv[3];

    int ret;
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    face_rec_result fr_result1;
    face_rec_result fr_result2;

    ret = init_arcface_model(model_path, &rknn_app_ctx);

    image_buffer_t src_image;
    memset(&src_image, 0, sizeof(image_buffer_t));
    ret = read_image(image1_path, &src_image);

    //RV1106 rga requires that input and output bufs are memory allocated by dma
    ret = dma_buf_alloc(RV1106_CMA_HEAP_PATH, src_image.size, &rknn_app_ctx.img_dma_buf.dma_buf_fd, 
                    (void **) & (rknn_app_ctx.img_dma_buf.dma_buf_virt_addr));
    memcpy(rknn_app_ctx.img_dma_buf.dma_buf_virt_addr, src_image.virt_addr, src_image.size);
    dma_sync_cpu_to_device(rknn_app_ctx.img_dma_buf.dma_buf_fd);
    free(src_image.virt_addr);
    src_image.virt_addr = (unsigned char *)rknn_app_ctx.img_dma_buf.dma_buf_virt_addr;
    src_image.fd = rknn_app_ctx.img_dma_buf.dma_buf_fd;
    rknn_app_ctx.img_dma_buf.size = src_image.size;

    ret = inference_arcface_model(&rknn_app_ctx, &src_image, &fr_result1);

    {
        image_buffer_t src_image;
        memset(&src_image, 0, sizeof(image_buffer_t));
        ret = read_image(image2_path, &src_image);

        //RV1106 rga requires that input and output bufs are memory allocated by dma
        ret = dma_buf_alloc(RV1106_CMA_HEAP_PATH, src_image.size, &rknn_app_ctx.img_dma_buf.dma_buf_fd, 
                        (void **) & (rknn_app_ctx.img_dma_buf.dma_buf_virt_addr));
        memcpy(rknn_app_ctx.img_dma_buf.dma_buf_virt_addr, src_image.virt_addr, src_image.size);
        dma_sync_cpu_to_device(rknn_app_ctx.img_dma_buf.dma_buf_fd);
        free(src_image.virt_addr);
        src_image.virt_addr = (unsigned char *)rknn_app_ctx.img_dma_buf.dma_buf_virt_addr;
        src_image.fd = rknn_app_ctx.img_dma_buf.dma_buf_fd;
        rknn_app_ctx.img_dma_buf.size = src_image.size;

        ret = inference_arcface_model(&rknn_app_ctx, &src_image, &fr_result2);
    }

    for (int i = 0; i < 512; i++)
    {
        printf("%f ", fr_result1.embedding[i]);
    }

    printf("\n");

    for (int i = 0; i < 512; i++)
    {
        printf("%f ", fr_result2.embedding[i]);
    }

    printf("\n");

    float similarity = cosineSimilarity<512>(fr_result1.embedding, fr_result2.embedding);

    printf("similarity: %f\n", similarity);     

    // printf("face_rec_result: ");
    // for (int i = 0; i < 512; i++)
    // {
    //     printf("%f ", fr_results.embedding[i]);
    // }

out:

    ret = release_arcface_model(&rknn_app_ctx);
    if (ret != 0)
    {
        printf("release_arcfacea_model fail! ret=%d\n", ret);
    }

    if (src_image.virt_addr != NULL)
    {
        dma_buf_free(rknn_app_ctx.img_dma_buf.size, &rknn_app_ctx.img_dma_buf.dma_buf_fd, 
                rknn_app_ctx.img_dma_buf.dma_buf_virt_addr);
    }

    return 0;
}
