<script lang="ts">
    import { page } from "$app/state";
    import AppIcon from "$lib/components/AppIcon.svelte";
    import Avatar from "$lib/components/Avatar.svelte";
    import Closure, { mkClosure, type SvClosure } from "$lib/components/Closure.svelte";
    import { mkSnippetClosure } from "$lib/components/SnippetClosure.svelte";
    import { HandCoins, House, Icon, ScrollText, User, Users } from "lucide-svelte";

    let { children, data } = $props();

    function mkIcon(icon: typeof Icon) {
        return mkSnippetClosure(renderIcon, icon);
    }

    const links: {
        icon: (active: boolean) => SvClosure;
        href: string;
        space?: boolean;
        large?: boolean;
    }[] = [
        { icon: () => mkIcon(House), href: "/dashboard" },
        { icon: () => mkIcon(HandCoins), href: "/issueloan" },
        { icon: () => mkIcon(ScrollText), href: "/history" },
        { icon: () => mkIcon(User), href: "/friends" },
        { icon: () => mkIcon(Users), href: "/groups" },
        {
            icon: () => mkClosure(Avatar, { user: data.user }),
            href: "/settings",
            space: true,
            large: true
        }
    ];
</script>

{#snippet renderIcon(icon: typeof Icon)}
    <AppIcon {icon} class="size-10" />
{/snippet}

<div class="grid h-full grid-rows-[auto_1fr]">
    <nav
        class="grid h-16 bg-neutral-800 text-neutral-100 shadow-lg dark:bg-neutral-100 dark:text-neutral-800"
    >
        <ul class="flex">
            {#each links as link}
                {@const active = page.url.pathname.startsWith(link.href)}
                <li class={["group grid flex-none", link.space && "ml-auto"]}>
                    <a
                        class={[
                            "relative",
                            active && "text-blue-500",
                            link.large
                                ? "px-[2px] py-2 group-first:pl-2 group-last:pr-2"
                                : "px-[6px] py-3 group-first:pl-3 group-last:pr-3"
                        ]}
                        href={link.href}
                    >
                        <Closure this={link.icon(active)} />
                        {#if active}
                            <div
                                class="absolute bottom-0 left-0 right-0 h-1 bg-current group-first:left-[6px] group-last:right-[6px]"
                            ></div>
                        {/if}
                    </a>
                </li>
            {/each}
        </ul>
    </nav>
    <div class="flex flex-col items-stretch gap-6 overflow-hidden p-6">{@render children()}</div>
</div>
