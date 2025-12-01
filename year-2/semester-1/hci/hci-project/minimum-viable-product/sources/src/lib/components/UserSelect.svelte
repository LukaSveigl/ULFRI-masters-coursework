<script lang="ts">
    import Avatar from "$lib/components/Avatar.svelte";
    import { Checkbox } from "$lib/shadcn/components/ui/checkbox";
    import { Input } from "$lib/shadcn/components/ui/input";
    import { sortUsers } from "$lib/sort";
    import type { OverrideSvelteHTMLElement } from "$lib/utils.svelte";
    import { SvelteSet } from "svelte/reactivity";

    let {
        selected = $bindable(new SvelteSet()),
        users,
        disabled = false,
        class: klass,
        ...props
    }: OverrideSvelteHTMLElement<
        "div",
        {
            selected?: SvelteSet<string>;
            users: {
                id: string;
                displayName: string;
                avatar: boolean;
                avatarModified: Date;
                username: string;
            }[];
            disabled?: boolean;
        }
    > = $props();

    let filter = $state("");
    let filterLower = $derived(filter.toLowerCase());
    let filteredUsers = $derived(
        users
            .filter((x) => {
                if (filter == "") return true;
                if (filter == filterLower) {
                    return x.displayName.toLowerCase().includes(filterLower);
                }
                return x.displayName.includes(filter);
            })
            .toSorted(sortUsers)
    );
</script>

<div {...props} class={["space-y-4", klass]}>
    <div>
        <Input type="search" placeholder="Search" {disabled} bind:value={filter} />
    </div>
    <ul class="flex flex-col">
        {#each filteredUsers as user (user.id)}
            <li class="group contents">
                <label
                    class="flex cursor-pointer items-center gap-4 border-t border-inherit py-2 group-last:border-b"
                >
                    <Checkbox
                        {disabled}
                        bind:checked={() => selected.has(user.id),
                        (x) => {
                            if (x) {
                                selected.add(user.id);
                            } else {
                                selected.delete(user.id);
                            }
                        }}
                    />
                    <div class="flex items-center gap-4 self-stretch">
                        <Avatar {user} />
                        <p>{user.displayName}</p>
                    </div>
                </label>
            </li>
        {/each}
    </ul>
</div>
