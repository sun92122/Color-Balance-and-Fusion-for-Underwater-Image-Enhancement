<template>
  <div style="justify-content: center; text-align: center">
    <h1>UIEB Sample Images</h1>
    <p>
      These images are samples from the
      <a href="https://li-chongyi.github.io/proj_benchmark.html">UIEB dataset</a
      >.
    </p>
    <div v-for="(image, index) in UIEB_sample_list" :key="index">
      <div class="card flex justify-center">
        <ImageCompare class="shadow-lg rounded-2xl">
          <template #right>
            <img :src="'./UIEB/' + image + '.jpg'" />
          </template>
          <template #left>
            <img :src="'./UIEB/' + image + '_result.jpg'" />
          </template>
        </ImageCompare>
      </div>
    </div>

    <h1>Paper Sample Images</h1>
    <p>
      These images are samples from the
      <a href="https://ieeexplore.ieee.org/document/8058463">paper</a>.
    </p>
    <div v-for="(image, index) in paper_sample_list" :key="index">
      <div class="card flex justify-center">
        <ImageCompare class="shadow-lg rounded-2xl">
          <template #right>
            <img :src="'./paper_sample/' + image + '.jpg'" />
          </template>
          <template #left>
            <img :src="'./paper_sample/' + image + '_result.jpg'" />
          </template>
        </ImageCompare>
      </div>
    </div>
  </div>
</template>

<script setup>
import ImageCompare from "primevue/imagecompare";

const UIEB_sample_list = [
  "4_img_",
  "3_img_",
  "6_img_",
  "7_img_",
  "13_img_",
  "15_img_",
  "19_img_",
  "28_img_",
  "30_img_",
  "32_img_",
  "37_img_",
  "86_img_",
  "87_img_",
  "105_img_",
  "169_img_",
  "910_img_",
  "8046",
  "9557",
  "9567",
];

const paper_sample_list = [
  "underwater_sample",
  "underwater_sample1",
  "underwater_sample2",
  "underwater_sample3",
  "underwater_sample4",
  "underwater_sample5",
];
</script>
