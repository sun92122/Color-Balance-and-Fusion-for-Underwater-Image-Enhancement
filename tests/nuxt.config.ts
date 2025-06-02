import Aura from "@primeuix/themes/aura";

export default defineNuxtConfig({
  compatibilityDate: "2025-05-15",
  devtools: { enabled: false },
  app: {
    baseURL: "/Color-Balance-and-Fusion-for-Underwater-Image-Enhancement/",
  },
  modules: ["@primevue/nuxt-module"],
  css: ["primeicons/primeicons.css"],
  nitro: {
    preset: "static",
    prerender: {
      autoSubfolderIndex: false,
    },
  },
  router: {
    options: {
      strict: true,
    },
  },
  primevue: {
    options: {
      theme: {
        preset: Aura,
        options: {
          dark: false, // Set to true for dark mode
          darkModeSelector: ".dark-mode-toggle",
        },
      },
    },
    components: {
      prefix: "Prime",
      exclude: ["Editor", "Chart", "Form", "FormField"], // To fix import error from PrimeVue
    },
  },

  ssr: false,
});
