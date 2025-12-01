<script lang="ts">
    import { invalidateAll } from "$app/navigation";
    import AppIcon from "$lib/components/AppIcon.svelte";
    import Avatar from "$lib/components/Avatar.svelte";
    import Label from "$lib/components/Label.svelte";
    import Popup from "$lib/components/Popup.svelte";
    import PopupDetailsGroup from "$lib/components/PopupDetailsGroup.svelte";
    import UserSelect from "$lib/components/UserSelect.svelte";
    import { Button, buttonVariants } from "$lib/shadcn/components/ui/button";
    import * as Dialog from "$lib/shadcn/components/ui/dialog";
    import { Input } from "$lib/shadcn/components/ui/input";
    import { ScrollArea } from "$lib/shadcn/components/ui/scroll-area";
    import { sortGroups } from "$lib/sort";
    import { mutation } from "$lib/utils.svelte.js";
    import { encodeBase64urlNoPadding } from "@oslojs/encoding";
    import { Plus } from "lucide-svelte";
    import { SvelteSet } from "svelte/reactivity";
    import { onCreateGroup, onRequestFriends } from "./page.telefunc";

    let { data } = $props();

    let filter = $state("");
    let filterLower = $derived(filter.toLowerCase());
    let filteredGroups = $derived(
        data.groups
            .filter((x) => {
                if (filter == "") return true;
                if (filter == filterLower) {
                    return x.displayName.toLowerCase().includes(filterLower);
                }
                return x.displayName.includes(filter);
            })
            .toSorted(sortGroups)
    );

    let detailsDialogOpen = $state(false);
    let detailsDialogGroup: (typeof data.groups)[number] | null = $state(null);

    let createGroupOpen = $state(false);
    let createGroupDisplayName = $state("");
    let createGroupInput: HTMLInputElement | null = $state(null);
    let createGroupFiles: FileList | null = $state(null);
    let createGroupUserIds = $state(new SvelteSet<string>());
    let createGroupError: string | null = $state(null);

    let createGroupAvatar: string | false = $state(false);
    $effect.pre(() => {
        if (createGroupFiles?.length != 1) {
            createGroupAvatar = false;
            return;
        }
        const url = URL.createObjectURL(createGroupFiles[0]);
        createGroupAvatar = url;
        return () => URL.revokeObjectURL(url);
    });

    const mutCreateGroup = mutation(
        async (config: { displayName: string; avatar: File | null; userIds: string[] }) => {
            const { ok } = await onCreateGroup({
                displayName: config.displayName,
                avatar: config.avatar
                    ? encodeBase64urlNoPadding(new Uint8Array(await config.avatar.arrayBuffer()))
                    : null,
                userIds: config.userIds
            });
            if (ok) {
                createGroupError = null;
                await invalidateAll();
                createGroupOpen = false;
            } else {
                createGroupError = "";
            }
        }
    );

    let createGroupFullErr = $derived(
        mutCreateGroup.stalled == "error" ? String(mutCreateGroup.stallError) : createGroupError
    );
</script>

<div class="relative h-full">
    <ScrollArea
        orientation="vertical"
        class="flex h-full flex-col items-stretch rounded-md border-2 border-neutral-300"
    >
        <div class="m-4">
            <Input type="search" placeholder="Search" bind:value={filter} />
        </div>
        <ul class="mx-4 mt-2 flex flex-col">
            {#each filteredGroups as group (group.id)}
                <li class="group contents">
                    <button
                        class="flex items-center gap-4 border-t border-inherit py-2 group-last:border-b"
                        onclick={() => {
                            detailsDialogOpen = true;
                            detailsDialogGroup = group;
                        }}
                    >
                        <Avatar user={group} isGroup />
                        <p>{group.displayName}</p>
                    </button>
                </li>
            {/each}
        </ul>
        <div class="h-16 my-2"></div>
    </ScrollArea>

    <Popup
        bind:open={createGroupOpen}
        onOpenChange={(open) => {
            if (open) {
                createGroupDisplayName = "New Group";
                createGroupFiles = null;
                createGroupUserIds.clear();
                createGroupError = null;
            }
        }}
    >
        {#snippet trigger()}
            <Dialog.Trigger
                class={[
                    buttonVariants(),
                    "absolute bottom-4 right-4 !size-12 !rounded-full shadow-md"
                ]}
            >
                <AppIcon icon={Plus} class="!size-6" />
            </Dialog.Trigger>
        {/snippet}
        <Dialog.Header>
            <Dialog.Title>Create Group</Dialog.Title>
        </Dialog.Header>
        <form
            onsubmit={(e) => {
                e.preventDefault();
                mutCreateGroup.desiredValue = {
                    displayName: createGroupDisplayName,
                    avatar: createGroupFiles?.length == 1 ? createGroupFiles[0] : null,
                    userIds: [...createGroupUserIds]
                };
            }}
        >
            <div class="grid gap-4 py-4">
                <div class="grid grid-cols-[auto_auto_1fr] grid-rows-[auto_auto] gap-x-4">
                    <input
                        class="hidden"
                        type="file"
                        accept="image/png"
                        bind:this={createGroupInput}
                        bind:files={createGroupFiles}
                        disabled={mutCreateGroup.applying}
                    />
                    <Avatar
                        class="row-start-2 cursor-pointer"
                        user={{
                            id: null,
                            avatar: createGroupAvatar,
                            displayName: createGroupDisplayName
                        }}
                        interactionReminder={!mutCreateGroup.applying}
                        onclick={() => {
                            createGroupInput?.click();
                        }}
                    />
                    <Label
                        class="col-span-2 col-start-2 row-span-2 grid grid-flow-col grid-cols-subgrid grid-rows-subgrid items-center"
                        label="Name"
                    >
                        <Input
                            class="flex-none justify-self-stretch"
                            disabled={mutCreateGroup.applying}
                            placeholder={data.user.username}
                            bind:value={createGroupDisplayName}
                        />
                    </Label>
                </div>
                <div>
                    <p class="mb-1 text-sm font-medium leading-none">Add Users</p>
                    <ScrollArea
                        orientation="vertical"
                        class="flex h-96 flex-col items-stretch rounded-md border-[1px] border-inherit"
                    >
                        {#await onRequestFriends() then friends}
                            <UserSelect
                                class="m-4"
                                bind:selected={createGroupUserIds}
                                users={friends}
                                disabled={mutCreateGroup.applying}
                            />
                        {/await}
                    </ScrollArea>
                </div>
            </div>
            {#if createGroupFullErr != null}
                <p class="text-sm text-destructive">{createGroupFullErr}</p>
            {/if}
            <Dialog.Footer>
                <Button
                    class={["shadow-md", mutCreateGroup.applying && "animate-pulse"]}
                    disabled={mutCreateGroup.applying}
                    type="submit"
                >
                    Create
                </Button>
            </Dialog.Footer>
        </form>
    </Popup>
</div>

<PopupDetailsGroup
    bind:open={detailsDialogOpen}
    group={detailsDialogGroup!}
    onleavegroup={async () => {
        await invalidateAll();
        detailsDialogOpen = false;
    }}
/>
