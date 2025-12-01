<script lang="ts">
    import type { OverrideComponentProps } from "$lib/utils.svelte";
    import { Icon, type IconNode } from "lucide-svelte";
    import type { Snippet } from "svelte";

    let { children, ...props }: OverrideComponentProps<Icon, { children?: Snippet<[]> }> = $props();

    const iconNode: IconNode = [
        ["path", { stroke: "none", d: "M0 0h24v24H0z", fill: "none" }],
        [
            "path",
            {
                d: "M9.5 13.75C9.5 14.72 10.25 15.5 11.17 15.5H13.05C13.85 15.5 14.5 14.82 14.5 13.97C14.5 13.06 14.1 12.73 13.51 12.52L10.5 11.47C9.91 11.26 9.51001 10.94 9.51001 10.02C9.51001 9.18 10.16 8.49001 10.96 8.49001H12.84C13.76 8.49001 14.51 9.27001 14.51 10.24M12 7.5V16.5M22 12C22 17.52 17.52 22 12 22C6.48 22 2 17.52 2 12C2 6.48 6.48 2 12 2M17 3V7M17 7H21M17 7L22 2",
                "stroke-linecap": "round",
                "stroke-linejoin": "round"
            }
        ]
    ];
</script>

<Icon name="money-in" {...props} {iconNode}>
    {@render children?.()}
</Icon>
