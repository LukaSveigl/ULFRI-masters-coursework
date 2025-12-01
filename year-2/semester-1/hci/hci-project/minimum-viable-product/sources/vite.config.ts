import { sveltekit } from "@sveltejs/kit/vite";
import { telefunc } from "telefunc/vite";
import { defineConfig } from "vite";

export default defineConfig({
    plugins: [sveltekit(), telefunc()]
});
