<script lang="ts">
    import type { OverrideComponentProps } from "$lib/utils.svelte";
    import type { Icon } from "lucide-svelte";
    import type { Component, ComponentProps } from "svelte";

    let {
        icon: ThisIcon,
        ...props
    }: OverrideComponentProps<
        Icon,
        {
            icon: Component<ComponentProps<Icon>> | typeof Icon;
        }
    > = $props();
</script>

<ThisIcon strokeWidth="1" {...props} />
